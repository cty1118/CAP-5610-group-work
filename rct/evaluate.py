"""Shared evaluation: the same metrics, files and summary table for every model.

In Python (after training):

    from rct.evaluate import save_run
    save_run("cnn_glove", split="validation", y_true=y, y_pred=p,
             config=vars(args), extra={"params": n_params, "train_seconds": t})

    -> results/cnn_glove/validation_metrics.json
       results/cnn_glove/validation_predictions.csv   (idx, true, pred)
       results/cnn_glove/validation_confusion.png

From the command line:

    # Score a prediction file made anywhere (Colab BERT run, an LLM, ...).
    # CSV columns: idx (row index in the split) and pred (label name or id).
    # A file may cover only part of the split (e.g. an LLM run on a test sample).
    python -m rct.evaluate score --pred-file llm_test.csv --split test --name llm_zero_shot

    # One table with every run in results/
    python -m rct.evaluate summary
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from rct.data import LABEL2ID, LABELS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def compute_metrics(y_true, y_pred) -> dict:
    """Macro-F1 (primary), accuracy, weighted-F1, per-class scores, confusion matrix."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(LABELS)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "n": int(len(y_true)),
        "macro_f1": round(100 * f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0), 2),
        "accuracy": round(100 * accuracy_score(y_true, y_pred), 2),
        "weighted_f1": round(100 * f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0), 2),
        "per_class": {
            lab: {
                "precision": round(100 * p[i], 2),
                "recall": round(100 * r[i], 2),
                "f1": round(100 * f[i], 2),
                "support": int(s[i]),
            }
            for i, lab in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def macro_f1(y_true, y_pred) -> float:
    """Shortcut for early stopping."""
    return 100 * f1_score(y_true, y_pred, average="macro", labels=list(range(len(LABELS))), zero_division=0)


def plot_confusion(cm, title: str, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    names = [lab.title() for lab in LABELS]
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{100 * norm[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "#14213D")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="Share of true class")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def save_run(name: str, split: str, y_true, y_pred, indices=None, config: dict = None,
             extra: dict = None, results_dir=RESULTS_DIR, plot: bool = True) -> dict:
    """Write metrics JSON, prediction CSV and confusion plot for one run on one split."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    indices = np.arange(len(y_true)) if indices is None else np.asarray(indices)
    out_dir = Path(results_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(y_true, y_pred)
    record = {"name": name, "split": split, **metrics, "config": config or {}, "extra": extra or {}}
    (out_dir / f"{split}_metrics.json").write_text(json.dumps(record, indent=2))

    with open(out_dir / f"{split}_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "true", "pred"])
        for i, t, p in zip(indices, y_true, y_pred):
            w.writerow([int(i), LABELS[t], LABELS[p]])

    if plot:
        plot_confusion(metrics["confusion_matrix"], f"{name} ({split}, macro-F1 {metrics['macro_f1']})",
                       out_dir / f"{split}_confusion.png")
    print(f"[{name} | {split}] macro-F1 {metrics['macro_f1']}  acc {metrics['accuracy']}  "
          f"weighted-F1 {metrics['weighted_f1']}  (n={metrics['n']:,})  -> {out_dir}")
    return record


def read_prediction_file(path) -> tuple:
    """(indices, predicted ids) from a CSV with columns idx and pred."""
    idx, pred = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            idx.append(int(row["idx"]))
            p = row["pred"].strip()
            pred.append(int(p) if p.isdigit() else LABEL2ID[p.upper()])
    return np.array(idx), np.array(pred)


def summarize(results_dir=RESULTS_DIR) -> str:
    """Markdown table of every *_metrics.json under results_dir."""
    rows = []
    for path in sorted(Path(results_dir).glob("*/*_metrics.json")):
        r = json.loads(path.read_text())
        e = r.get("extra", {})
        rows.append([
            r["name"], r["split"], f"{r['n']:,}", r["macro_f1"], r["accuracy"], r["weighted_f1"],
            *(r["per_class"][lab]["f1"] for lab in LABELS),
            f"{e['params']:,}" if "params" in e else "",
            f"{e['train_seconds'] / 60:.1f}" if "train_seconds" in e else "",
        ])
    head = ["Run", "Split", "n", "Macro-F1", "Acc", "W-F1", *(lab[:4].title() for lab in LABELS),
            "Params", "Train min"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="score a prediction CSV (idx, pred)")
    s.add_argument("--pred-file", required=True)
    s.add_argument("--split", required=True, choices=["train", "validation", "test"])
    s.add_argument("--name", required=True)
    s.add_argument("--results-dir", default=RESULTS_DIR)

    m = sub.add_parser("summary", help="print a table of all runs")
    m.add_argument("--results-dir", default=RESULTS_DIR)
    m.add_argument("--out", help="also write the table to this .md file")

    args = ap.parse_args()
    if args.cmd == "score":
        from rct.data import load_splits

        idx, pred = read_prediction_file(args.pred_file)
        y_true = load_splits()[args.split].labels[idx]
        save_run(args.name, args.split, y_true, pred, indices=idx,
                 config={"pred_file": str(args.pred_file)}, results_dir=args.results_dir)
    else:
        table = summarize(args.results_dir)
        print(table)
        if args.out:
            Path(args.out).write_text(table + "\n")


if __name__ == "__main__":
    main()
