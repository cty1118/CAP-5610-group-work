"""Train TextCNN or BiLSTM on PubMed 20k RCT (Methods 1 and 2).

Tunes on validation only (early stopping on validation macro-F1). The test set
is scored only when --eval-test is passed: do that once, with the final config.

Examples:
    python train_nn.py --model cnn                                # proposal start config
    python train_nn.py --model lstm --pooling attention
    python train_nn.py --model cnn --embedding glove --glove-path glove.6B.300d.txt
    python train_nn.py --model cnn --no-position                  # position ablation
    python train_nn.py --model cnn --subset 5000 --epochs 1       # quick smoke test
    python train_nn.py --model cnn --name cnn_final --eval-test   # final run, touches test

Outputs go to results/<name>/ (see rct/evaluate.py); a summary of all runs:
    python -m rct.evaluate summary
"""

import argparse
import copy
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rct.data import (
    DEFAULT_MAX_LEN, POS_DIM, build_vocab, encode, load_glove, load_splits, position_features,
)
from rct.evaluate import RESULTS_DIR, macro_f1, save_run
from rct.models import BiLSTM, TextCNN, count_params

MODEL_DEFAULTS = {  # proposal "Start" column; None in args means "use this"
    "cnn": {"dropout": 0.5, "clip": 0.0},
    "lstm": {"dropout": 0.3, "clip": 1.0},
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["cnn", "lstm"])
    ap.add_argument("--name", help="run name (default: built from the hyperparameters)")
    # data
    ap.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    ap.add_argument("--min-freq", type=int, default=2, help="min train count to enter the vocabulary")
    ap.add_argument("--no-position", action="store_true", help="ablation: drop position features")
    ap.add_argument("--subset", type=int, default=0, help="use N train sentences (and N/5 val/test) for a quick test")
    # embeddings
    ap.add_argument("--embedding", choices=["random", "glove"], default="random")
    ap.add_argument("--glove-path", help="glove.6B.300d.txt (needed for --embedding glove)")
    ap.add_argument("--emb-dim", type=int, default=128, help="ignored for glove (300)")
    ap.add_argument("--freeze-emb", action="store_true")
    # CNN
    ap.add_argument("--filter-widths", default="3,4,5")
    ap.add_argument("--n-filters", type=int, default=100)
    # LSTM
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--pooling", choices=["last", "attention"], default="last")
    # training
    ap.add_argument("--dropout", type=float, default=None, help="default 0.5 for cnn, 0.3 for lstm")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=2, help="epochs without val macro-F1 gain before stopping")
    ap.add_argument("--clip", type=float, default=None, help="grad-norm clip, 0 = off (default 1.0 for lstm)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    # outputs
    ap.add_argument("--eval-test", action="store_true", help="also score the test set (final run only)")
    ap.add_argument("--save-model", action="store_true", help="save the best weights to checkpoints/<name>.pt")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = ap.parse_args(argv)

    for k, v in MODEL_DEFAULTS[args.model].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.embedding == "glove":
        if not args.glove_path:
            ap.error("--embedding glove needs --glove-path")
        args.emb_dim = 300
    if not args.name:
        args.name = default_name(args)
    return args


def default_name(a) -> str:
    emb = "glove" if a.embedding == "glove" else f"rand{a.emb_dim}"
    if a.model == "cnn":
        arch = f"w{a.filter_widths.replace(',', '')}_f{a.n_filters}"
    else:
        arch = f"h{a.hidden}_l{a.layers}_{a.pooling}"
    name = f"{a.model}_{emb}_{arch}_d{a.dropout}_lr{a.lr:g}"
    if a.no_position:
        name += "_nopos"
    if a.subset:
        name += f"_sub{a.subset}"
    return name


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_tensors(split, vocab, max_len, use_pos):
    ids, lengths = encode(split.texts, vocab, max_len)
    pos = position_features(split) if use_pos else np.zeros((len(split), 0), dtype=np.float32)
    return TensorDataset(torch.from_numpy(ids), torch.from_numpy(lengths),
                         torch.from_numpy(pos), torch.from_numpy(split.labels))


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    preds = []
    for ids, lengths, pos, _ in loader:
        logits = model(ids.to(device), lengths, pos.to(device))
        preds.append(logits.argmax(dim=1).cpu())
    return torch.cat(preds).numpy()


def build_model(args, vocab):
    pretrained = load_glove(args.glove_path, vocab, dim=300, seed=args.seed) if args.embedding == "glove" else None
    common = dict(vocab_size=len(vocab), emb_dim=args.emb_dim, dropout=args.dropout,
                  pos_dim=0 if args.no_position else POS_DIM, pretrained=pretrained, freeze_emb=args.freeze_emb)
    if args.model == "cnn":
        widths = tuple(int(w) for w in args.filter_widths.split(","))
        return TextCNN(filter_widths=widths, n_filters=args.n_filters, **common)
    return BiLSTM(hidden=args.hidden, layers=args.layers, pooling=args.pooling, **common)


def run(argv=None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"== {args.name}  (device: {device})")

    splits = load_splits()
    if args.subset:
        splits = {k: v.subset(args.subset if k == "train" else max(args.subset // 5, 500), seed=args.seed)
                  for k, v in splits.items()}
    vocab = build_vocab(splits["train"].texts, min_freq=args.min_freq)
    use_pos = not args.no_position
    data = {k: make_tensors(v, vocab, args.max_len, use_pos) for k, v in splits.items()}
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(data["train"], batch_size=args.batch_size, shuffle=True, generator=g)
    eval_loaders = {k: DataLoader(data[k], batch_size=512) for k in ("validation", "test")}

    model = build_model(args, vocab).to(device)
    n_params = count_params(model)
    print(f"vocab {len(vocab):,}  train {len(splits['train']):,}  val {len(splits['validation']):,}  "
          f"trainable params {n_params:,}")
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, best_epoch, best_state, history, bad = -1.0, 0, None, [], 0
    y_val = splits["validation"].labels
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_seen = 0.0, 0
        for ids, lengths, pos, y in train_loader:
            ids, pos, y = ids.to(device), pos.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(ids, lengths, pos), y)
            loss.backward()
            if args.clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            total_loss += loss.item() * len(y)
            n_seen += len(y)
        val_f1 = macro_f1(y_val, predict(model, eval_loaders["validation"], device))
        history.append({"epoch": epoch, "train_loss": round(total_loss / n_seen, 4), "val_macro_f1": round(val_f1, 2)})
        print(f"epoch {epoch:2d}  train loss {total_loss / n_seen:.4f}  val macro-F1 {val_f1:.2f}  "
              f"({time.time() - t0:.0f}s)")
        if val_f1 > best_f1:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop: no gain for {args.patience} epochs")
                break
    train_seconds = time.time() - t0

    model.load_state_dict(best_state)
    if args.save_model:
        from pathlib import Path

        Path("checkpoints").mkdir(exist_ok=True)
        torch.save({"state_dict": best_state, "vocab": vocab, "config": vars(args)}, f"checkpoints/{args.name}.pt")

    extra = {"params": n_params, "train_seconds": round(train_seconds, 1), "best_epoch": best_epoch,
             "vocab_size": len(vocab), "device": str(device), "history": history}
    config = {k: v for k, v in vars(args).items() if k not in ("results_dir",)}
    out = {"validation": save_run(args.name, "validation", y_val, predict(model, eval_loaders["validation"], device),
                                  indices=splits["validation"].index, config=config, extra=extra,
                                  results_dir=args.results_dir)}
    if args.eval_test:
        out["test"] = save_run(args.name, "test", splits["test"].labels,
                               predict(model, eval_loaders["test"], device),
                               indices=splits["test"].index, config=config, extra=extra,
                               results_dir=args.results_dir)
    return out


if __name__ == "__main__":
    run()
