"""Train the MLP on TF-IDF features (Method 3) on PubMed 20k RCT.

Same splits, labels, position features, early stopping and result files as
train_nn.py, so the MLP sits in the same summary table as the CNN and BiLSTM.
Tunes on validation only; the test set is scored only with --eval-test.

Examples:
    python train_mlp.py                                    # proposal start config
    python train_mlp.py --hidden 1024,512 --dropout 0.3
    python train_mlp.py --no-position                      # position ablation
    python train_mlp.py --subset 5000 --epochs 1           # quick smoke test
    python train_mlp.py --name mlp_final --eval-test       # final run, touches test

Start config (proposal slide 10): TF-IDF 1-2-grams, 50k features, + 4 position
features -> Dense 512 -> Dense 256 -> 5 classes, ReLU, dropout 0.5, Adam 1e-3,
batch 256, early stopping on validation macro-F1 with patience 2.
"""

import argparse
import copy
import random
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from rct.data import POS_DIM, load_splits, position_features, tfidf_features
from rct.evaluate import RESULTS_DIR, macro_f1, save_run
from rct.mlp import TfidfMLP, csr_batch
from rct.models import count_params

# TF-IDF fits take ~1 min; a sweep reuses them across runs in one process.
_TFIDF_CACHE = {}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="run name (default: built from the hyperparameters)")
    # features
    ap.add_argument("--max-features", type=int, default=50_000, help="TF-IDF vocabulary size")
    ap.add_argument("--ngram-max", type=int, default=2, help="TF-IDF uses 1..N-grams")
    ap.add_argument("--no-position", action="store_true", help="ablation: drop position features")
    ap.add_argument("--subset", type=int, default=0, help="use N train sentences (and N/5 val/test) for a quick test")
    # model
    ap.add_argument("--hidden", default="512,256", help="hidden layer sizes; the count is the depth")
    ap.add_argument("--dropout", type=float, default=0.5)
    # training
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20, help="upper bound; early stopping usually ends sooner")
    ap.add_argument("--patience", type=int, default=2, help="epochs without val macro-F1 gain before stopping")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    # outputs
    ap.add_argument("--eval-test", action="store_true", help="also score the test set (final run only)")
    ap.add_argument("--save-model", action="store_true", help="save the best weights to checkpoints/<name>.pt")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = ap.parse_args(argv)
    if not args.name:
        args.name = default_name(args)
    return args


def default_name(a) -> str:
    name = f"mlp_h{a.hidden.replace(',', '-')}_d{a.dropout}_tf{a.max_features // 1000}k_lr{a.lr:g}"
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


def get_tfidf(splits, max_features, ngram_max, subset, seed):
    key = (max_features, ngram_max, subset, seed)
    if key not in _TFIDF_CACHE:
        t0 = time.time()
        X, vec = tfidf_features(splits, max_features=max_features, ngram_range=(1, ngram_max))
        X = {k: sp.csr_matrix(v, dtype=np.float32) for k, v in X.items()}
        print(f"TF-IDF: {len(vec.vocabulary_):,} features, fit in {time.time() - t0:.0f}s")
        _TFIDF_CACHE[key] = X
    return _TFIDF_CACHE[key]


@torch.no_grad()
def predict(model, X, pos, device, batch_size=4096) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, X.shape[0], batch_size):
        rows = np.arange(start, min(start + batch_size, X.shape[0]))
        logits = model(*csr_batch(X, rows, device), pos[rows].to(device))
        preds.append(logits.argmax(dim=1).cpu())
    return torch.cat(preds).numpy()


def run(argv=None) -> dict:
    args = parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"== {args.name}  (device: {device})")

    splits = load_splits()
    if args.subset:
        splits = {k: v.subset(args.subset if k == "train" else max(args.subset // 5, 500), seed=args.seed)
                  for k, v in splits.items()}
    X = get_tfidf(splits, args.max_features, args.ngram_max, args.subset, args.seed)
    use_pos = not args.no_position
    pos = {k: torch.from_numpy(position_features(v) if use_pos else np.zeros((len(v), 0), np.float32))
           for k, v in splits.items()}

    hidden = tuple(int(h) for h in args.hidden.split(","))
    model = TfidfMLP(X["train"].shape[1], hidden=hidden, dropout=args.dropout,
                     pos_dim=POS_DIM if use_pos else 0).to(device)
    n_params = count_params(model)
    print(f"tfidf dim {X['train'].shape[1]:,}  train {len(splits['train']):,}  val {len(splits['validation']):,}  "
          f"trainable params {n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    y_train = torch.from_numpy(splits["train"].labels)
    y_val = splits["validation"].labels
    rng = np.random.default_rng(args.seed)
    best_f1, best_epoch, best_state, history, bad = -1.0, 0, None, [], 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_seen = 0.0, 0
        order = rng.permutation(X["train"].shape[0])
        for start in range(0, len(order), args.batch_size):
            rows = order[start:start + args.batch_size]
            y = y_train[rows].to(device)
            opt.zero_grad()
            loss = loss_fn(model(*csr_batch(X["train"], rows, device), pos["train"][rows].to(device)), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(rows)
            n_seen += len(rows)
        val_f1 = macro_f1(y_val, predict(model, X["validation"], pos["validation"], device))
        history.append({"epoch": epoch, "train_loss": round(total_loss / n_seen, 4), "val_macro_f1": round(val_f1, 2)})
        print(f"epoch {epoch:2d}  train loss {total_loss / n_seen:.4f}  val macro-F1 {val_f1:.2f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
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
        torch.save({"state_dict": best_state, "config": vars(args)}, f"checkpoints/{args.name}.pt")

    extra = {"params": n_params, "train_seconds": round(train_seconds, 1), "best_epoch": best_epoch,
             "tfidf_dim": int(X["train"].shape[1]), "device": str(device), "history": history}
    config = {k: v for k, v in vars(args).items() if k not in ("results_dir",)}
    out = {"validation": save_run(args.name, "validation", y_val,
                                  predict(model, X["validation"], pos["validation"], device),
                                  indices=splits["validation"].index, config=config, extra=extra,
                                  results_dir=args.results_dir)}
    if args.eval_test:
        out["test"] = save_run(args.name, "test", splits["test"].labels,
                               predict(model, X["test"], pos["test"], device),
                               indices=splits["test"].index, config=config, extra=extra,
                               results_dir=args.results_dir)
    return out


if __name__ == "__main__":
    run()
