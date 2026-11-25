"""Fine-tune BiomedBERT (Method 4) or DistilBERT on PubMed 20k RCT.

Same splits, labels, position features and result files as the other models.
Each epoch is scored on validation; the epoch with the best validation macro-F1
is kept. The test set is scored only with --eval-test (final runs).

Examples:
    python train_bert.py                                         # BiomedBERT, proposal start config
    python train_bert.py --subset 30000 --lr 3e-5                # LR sweep on a 30k-sentence subset
    python train_bert.py --model distilbert                      # general-domain comparison
    python train_bert.py --no-position                           # position ablation
    python train_bert.py --subset 2000 --eval-subset 1000 --epochs 1   # smoke test (CPU is fine)
    python train_bert.py --name bert_final --eval-test           # final run, touches test

Start config (proposal slide 11): WordPiece max_len 64, [CLS] vector + 4 position
features -> Linear -> 5 classes, AdamW lr 2e-5, 3 epochs, batch 32, 10% linear
warmup, weight decay 0.01. The new classification head gets its own learning
rate (--head-lr, default 1e-3): at 2e-5 the head would barely learn to use the
position features within 3 epochs.
"""

import argparse
import copy
import os
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn

from rct.data import NUM_CLASSES, POS_DIM, load_splits, position_features
from rct.evaluate import RESULTS_DIR, macro_f1, save_run

CHECKPOINTS = {
    # Gu et al. 2021 (formerly PubMedBERT): BERT-base, vocabulary and weights learned from PubMed
    "biomedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    # general-domain comparison (Wikipedia + books), 6 layers
    "distilbert": "distilbert-base-uncased",
    # general-domain BERT-base, same 12-layer size as BiomedBERT: isolates the effect of PubMed pretraining
    "bertbase": "bert-base-uncased",
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="run name (default: built from the hyperparameters)")
    ap.add_argument("--model", choices=list(CHECKPOINTS), default="biomedbert")
    ap.add_argument("--checkpoint", help="Hugging Face model id or local path (overrides --model)")
    # data
    ap.add_argument("--max-len", type=int, default=64, help="WordPiece tokens per sentence, incl. [CLS]/[SEP]")
    ap.add_argument("--no-position", action="store_true", help="ablation: drop position features")
    ap.add_argument("--subset", type=int, default=0, help="train on N random train sentences (0 = all)")
    ap.add_argument("--eval-subset", type=int, default=0, help="score N random val/test sentences (0 = all)")
    # training
    ap.add_argument("--lr", type=float, default=2e-5, help="encoder learning rate")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="learning rate of the new linear head")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--warmup", type=float, default=0.1, help="share of steps with linear warmup")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1, help="dropout before the head")
    ap.add_argument("--clip", type=float, default=1.0, help="grad-norm clip, 0 = off")
    ap.add_argument("--patience", type=int, default=0, help="stop after N epochs without val gain (0 = run all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--amp", choices=["auto", "bf16", "fp16", "off"], default="auto",
                    help="mixed precision on CUDA: bf16 where supported (H100), else fp16 (V100)")
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--log-every", type=int, default=200, help="print training progress every N steps")
    # outputs
    ap.add_argument("--eval-test", action="store_true", help="also score the test set (final run only)")
    ap.add_argument("--save-model", action="store_true", help="save the best weights to checkpoints/<name>.pt")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = ap.parse_args(argv)
    args.checkpoint = args.checkpoint or CHECKPOINTS[args.model]
    if not args.name:
        args.name = default_name(args)
    return args


def default_name(a) -> str:
    name = f"bert_{a.model}_lr{a.lr:g}_e{a.epochs}_len{a.max_len}"
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
    torch.cuda.manual_seed_all(seed)


class BertSentenceClassifier(nn.Module):
    """[CLS] vector of a pretrained encoder, concatenated with the position features, -> Linear."""

    def __init__(self, checkpoint, pos_dim=0, dropout=0.1, num_classes=NUM_CLASSES):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size + pos_dim, num_classes)

    def forward(self, input_ids, attention_mask, pos):
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(self.dropout(torch.cat([cls, pos.to(cls.dtype)], dim=1)))


class Encoded:
    """Token ids (unpadded), position features and labels for one split."""

    def __init__(self, split, tokenizer, max_len, use_pos):
        enc = tokenizer(split.texts, truncation=True, max_length=max_len, padding=False,
                        return_attention_mask=False, return_token_type_ids=False)
        self.ids = enc["input_ids"]
        self.lengths = np.array([len(x) for x in self.ids])
        self.pos = position_features(split) if use_pos else np.zeros((len(split), 0), np.float32)
        self.labels = split.labels
        # share of sentences cut at max_len, estimated on the first 5,000
        self.truncated = float(np.mean(
            [len(t) > max_len for t in tokenizer(split.texts[:5000], add_special_tokens=True)["input_ids"]]))

    def __len__(self):
        return len(self.ids)

    def batch(self, rows, pad_id, device):
        width = int(self.lengths[rows].max())
        ids = np.full((len(rows), width), pad_id, dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.int64)
        for j, r in enumerate(rows):
            n = self.lengths[r]
            ids[j, :n] = self.ids[r]
            mask[j, :n] = 1
        return (torch.from_numpy(ids).to(device, non_blocking=True),
                torch.from_numpy(mask).to(device, non_blocking=True),
                torch.from_numpy(self.pos[rows]).to(device, non_blocking=True))


def autocast_ctx(device, amp):
    if device.type != "cuda" or amp == "off":
        return nullcontext(), None
    if amp == "auto":
        # native bf16 needs compute capability 8.0+ (A100/H100); is_bf16_supported() also
        # says yes on a V100, where bf16 is only emulated and several times slower than fp16
        amp = "bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16"
    if amp == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16), None
    return torch.autocast("cuda", dtype=torch.float16), torch.amp.GradScaler("cuda")


@torch.no_grad()
def predict(model, data, pad_id, device, amp, batch_size) -> np.ndarray:
    model.eval()
    # length-sorted batches waste less padding; predictions are put back in split order
    order = np.argsort(data.lengths, kind="stable")
    preds = np.empty(len(data), dtype=np.int64)
    for start in range(0, len(order), batch_size):
        rows = order[start:start + batch_size]
        ctx, _ = autocast_ctx(device, amp)
        with ctx:
            logits = model(*data.batch(rows, pad_id, device))
        preds[rows] = logits.argmax(dim=1).cpu().numpy()
    return preds


def run(argv=None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    if device.type == "cuda" and args.amp == "auto":
        dev_name += " (bf16)" if torch.cuda.get_device_capability(device)[0] >= 8 else " (fp16)"
    print(f"== {args.name}  (device: {dev_name}, checkpoint: {args.checkpoint})", flush=True)

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    splits = load_splits()
    if args.subset:
        splits["train"] = splits["train"].subset(args.subset, seed=args.seed)
    if args.eval_subset:
        for k in ("validation", "test"):
            splits[k] = splits[k].subset(args.eval_subset, seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    use_pos = not args.no_position
    t0 = time.time()
    data = {k: Encoded(v, tokenizer, args.max_len, use_pos) for k, v in splits.items()}
    print(f"tokenized in {time.time() - t0:.0f}s; train {len(data['train']):,}  val {len(data['validation']):,}  "
          f"mean tokens {data['train'].lengths.mean():.1f}  "
          f"truncated at {args.max_len}: {100 * data['train'].truncated:.1f}% of train", flush=True)

    model = BertSentenceClassifier(args.checkpoint, pos_dim=POS_DIM if use_pos else 0,
                                   dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    enc_named = list(model.encoder.named_parameters())
    groups = [
        {"params": [p for n, p in enc_named if not n.endswith(no_decay)], "weight_decay": args.weight_decay,
         "lr": args.lr},
        {"params": [p for n, p in enc_named if n.endswith(no_decay)], "weight_decay": 0.0, "lr": args.lr},
        {"params": list(model.head.parameters()), "weight_decay": args.weight_decay, "lr": args.head_lr},
    ]
    opt = torch.optim.AdamW(groups, lr=args.lr)
    steps_per_epoch = (len(data["train"]) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total_steps), total_steps)
    loss_fn = nn.CrossEntropyLoss()
    pad_id = tokenizer.pad_token_id
    print(f"trainable params {n_params:,}  steps/epoch {steps_per_epoch:,}  total steps {total_steps:,}", flush=True)

    y_train = torch.from_numpy(data["train"].labels)
    y_val = data["validation"].labels
    rng = np.random.default_rng(args.seed)
    best_f1, best_epoch, best_state, history, bad = -1.0, 0, None, [], 0
    t0 = time.time()
    step = 0
    _, scaler = autocast_ctx(device, args.amp)  # one GradScaler for the whole run (fp16 only)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_seen, t_ep = torch.zeros((), device=device), 0, time.time()
        order = rng.permutation(len(data["train"]))
        for start in range(0, len(order), args.batch_size):
            rows = order[start:start + args.batch_size]
            ids, mask, pos = data["train"].batch(rows, pad_id, device)
            y = y_train[rows].to(device)
            ctx, _ = autocast_ctx(device, args.amp)
            with ctx:
                loss = loss_fn(model(ids, mask, pos).float(), y)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            else:
                loss.backward()
            if args.clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            sched.step()
            step += 1
            total_loss += loss.detach() * len(rows)  # no .item() here: avoids a GPU sync every step
            n_seen += len(rows)
            if args.log_every and step % args.log_every == 0:
                rate = n_seen / (time.time() - t_ep)
                print(f"  epoch {epoch} step {step:,}/{total_steps:,}  loss {total_loss.item() / n_seen:.4f}  "
                      f"{rate:.0f} sent/s", flush=True)
        val_f1 = macro_f1(y_val, predict(model, data["validation"], pad_id, device, args.amp, args.eval_batch_size))
        total_loss = total_loss.item()
        history.append({"epoch": epoch, "train_loss": round(total_loss / n_seen, 4), "val_macro_f1": round(val_f1, 2),
                        "seconds": round(time.time() - t_ep, 1)})
        print(f"epoch {epoch:2d}  train loss {total_loss / n_seen:.4f}  val macro-F1 {val_f1:.2f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if val_f1 > best_f1:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                print(f"early stop: no gain for {args.patience} epochs")
                break
    train_seconds = time.time() - t0

    model.load_state_dict(best_state)
    if args.save_model:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({"state_dict": best_state, "config": vars(args)}, f"checkpoints/{args.name}.pt")

    extra = {"params": n_params, "train_seconds": round(train_seconds, 1), "best_epoch": best_epoch,
             "device": dev_name, "history": history, "train_size": len(data["train"]),
             "truncated_share": round(data["train"].truncated, 4)}
    config = {k: v for k, v in vars(args).items() if k not in ("results_dir",)}
    out = {"validation": save_run(args.name, "validation", y_val,
                                  predict(model, data["validation"], pad_id, device, args.amp, args.eval_batch_size),
                                  indices=splits["validation"].index, config=config, extra=extra,
                                  results_dir=args.results_dir)}
    if args.eval_test:
        out["test"] = save_run(args.name, "test", data["test"].labels,
                               predict(model, data["test"], pad_id, device, args.amp, args.eval_batch_size),
                               indices=splits["test"].index, config=config, extra=extra,
                               results_dir=args.results_dir)
    return out


if __name__ == "__main__":
    run()
