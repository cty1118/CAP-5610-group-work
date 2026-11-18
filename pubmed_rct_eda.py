"""Load PubMed 20k RCT from Hugging Face and run exploratory data analysis.

Usage:
    pip install datasets numpy matplotlib
    python pubmed_rct_eda.py                 # cache -> ./data/hf_cache, outputs -> ./eda_output
    python pubmed_rct_eda.py --max-len 55    # also report coverage for a chosen max_len

Outputs (in --out-dir):
    stats.json                      every number below, machine-readable
    summary.md                      tables ready to paste into slides / report
    class_distribution.png          sentences per class, per split
    sentence_length_hist.png        tokens per sentence (train) with p95 / p99 lines
    sentences_per_abstract.png      sentences per abstract (train)
    label_by_position.png           label share vs relative position in the abstract
    length_by_class.png             token-length boxplot per class (train)

Data source: Hugging Face dataset "armanc/pubmed-rct20k"
(https://huggingface.co/datasets/armanc/pubmed-rct20k), the cleaned version
(Jin & Szolovits, 2018) of PubMed 20k RCT (Dernoncourt & Lee, IJCNLP 2017).
Numbers in the text are already replaced by "@".
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

HF_DATASET = "armanc/pubmed-rct20k"
SPLITS = ["train", "validation", "test"]
LABELS = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]

# Palette shared with the proposal slides.
BLUE, ORANGE, INK, GREY = "#1F5FA8", "#D9822B", "#14213D", "#8A97A8"
SPLIT_COLORS = {"train": BLUE, "validation": ORANGE, "test": GREY}


def load_records(cache_dir: Path) -> dict:
    """Load all splits from Hugging Face; return {split: list of sentence records}.

    Each Hugging Face row has abstract_id, sentence_id (0-based), label and text.
    total_lines (sentences in the abstract) is derived by counting per abstract_id.
    """
    ds = load_dataset(HF_DATASET, cache_dir=str(cache_dir))
    data = {}
    for split in SPLITS:
        rows = ds[split]
        abstract_ids, labels, texts, sentence_ids = (
            rows["abstract_id"], rows["label"], rows["text"], rows["sentence_id"])
        total = Counter(abstract_ids)
        data[split] = [
            {
                "pmid": pmid,
                "line_number": sid,
                "total_lines": total[pmid],
                "label": label.upper(),
                "text": text,
                "n_tokens": len(text.split()),
            }
            for pmid, label, text, sid in zip(abstract_ids, labels, texts, sentence_ids)
        ]
    return data


def describe(values) -> dict:
    a = np.asarray(values)
    return {
        "mean": round(float(a.mean()), 2),
        "std": round(float(a.std()), 2),
        "min": int(a.min()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": int(a.max()),
    }


def split_stats(records: list) -> dict:
    counts = Counter(r["label"] for r in records)
    n = len(records)
    per_abstract = Counter(r["pmid"] for r in records)
    return {
        "n_abstracts": len(per_abstract),
        "n_sentences": n,
        "class_counts": {lab: counts.get(lab, 0) for lab in LABELS},
        "class_percent": {lab: round(100 * counts.get(lab, 0) / n, 2) for lab in LABELS},
        "imbalance_ratio_max_to_min": round(max(counts.values()) / min(counts.values()), 2),
        "tokens_per_sentence": describe([r["n_tokens"] for r in records]),
        "sentences_per_abstract": describe(list(per_abstract.values())),
    }


def plot_class_distribution(stats: dict, out: Path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(LABELS))
    width = 0.27
    for i, split in enumerate(SPLITS):
        pct = [stats[split]["class_percent"][lab] for lab in LABELS]
        bars = ax.bar(x + (i - 1) * width, pct, width, label=split, color=SPLIT_COLORS[split])
        if split == "train":
            for b, lab in zip(bars, LABELS):
                ax.annotate(f"{stats[split]['class_counts'][lab]:,}",
                            (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [lab.title() for lab in LABELS])
    ax.set_ylabel("Share of sentences (%)")
    ax.set_title("Class distribution by split (counts shown for train)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_length_hist(lengths, desc: dict, max_len, out: Path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(lengths, bins=range(0, min(int(desc["max"]) + 2, 151)), color=BLUE, alpha=0.85)
    for key, color in (("p95", ORANGE), ("p99", INK)):
        ax.axvline(desc[key], color=color, linestyle="--", linewidth=1.5,
                   label=f"{key} = {desc[key]:.0f} tokens")
    if max_len:
        ax.axvline(max_len, color=GREY, linestyle=":", linewidth=1.5, label=f"max_len = {max_len}")
    ax.set_xlabel("Tokens per sentence (whitespace split, clipped at 150)")
    ax.set_ylabel("Sentences")
    ax.set_title(f"Sentence length, train (mean {desc['mean']}, median {desc['median']:.0f})")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_sentences_per_abstract(records: list, out: Path):
    per_abstract = list(Counter(r["pmid"] for r in records).values())
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(per_abstract, bins=range(0, max(per_abstract) + 2), color=BLUE, alpha=0.85)
    ax.set_xlabel("Sentences per abstract")
    ax.set_ylabel("Abstracts")
    ax.set_title(f"Sentences per abstract, train (mean {np.mean(per_abstract):.1f})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_label_by_position(records: list, out: Path, n_bins: int = 10):
    """Share of each label within each decile of relative position."""
    grid = np.zeros((len(LABELS), n_bins))
    for r in records:
        rel = r["line_number"] / max(r["total_lines"] - 1, 1)
        b = min(int(rel * n_bins), n_bins - 1)
        grid[LABELS.index(r["label"]), b] += 1
    grid = grid / grid.sum(axis=0, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(grid, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    for i in range(len(LABELS)):
        for j in range(n_bins):
            ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if grid[i, j] > 55 else INK)
    ax.set_yticks(range(len(LABELS)), [lab.title() for lab in LABELS])
    ax.set_xticks(range(n_bins), [f"{10 * j}–{10 * (j + 1)}%" for j in range(n_bins)], fontsize=8)
    ax.set_xlabel("Relative position of the sentence in its abstract")
    ax.set_title("Label share (%) by position, train: why the position feature helps")
    fig.colorbar(im, ax=ax, label="% of sentences in that position bin")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_length_by_class(records: list, out: Path):
    data = [[r["n_tokens"] for r in records if r["label"] == lab] for lab in LABELS]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.boxplot(data, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#DCE7F3", "edgecolor": BLUE},
               medianprops={"color": ORANGE, "linewidth": 2})
    ax.set_xticks(range(1, len(LABELS) + 1), [lab.title() for lab in LABELS])
    ax.set_ylabel("Tokens per sentence")
    ax.set_title("Sentence length by class, train (outliers hidden)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def write_summary(stats: dict, out: Path):
    lines = [f"# PubMed 20k RCT ({HF_DATASET}): EDA summary", "",
             "| Split | Abstracts | Sentences | Imbalance (max/min) |",
             "|---|---:|---:|---:|"]
    for split in SPLITS:
        s = stats[split]
        lines.append(f"| {split} | {s['n_abstracts']:,} | {s['n_sentences']:,} "
                     f"| {s['imbalance_ratio_max_to_min']} |")
    lines += ["", "## Class distribution", "",
              "| Class | " + " | ".join(SPLITS) + " |",
              "|---|" + "---:|" * len(SPLITS)]
    for lab in LABELS:
        cells = [f"{stats[s]['class_counts'][lab]:,} ({stats[s]['class_percent'][lab]}%)" for s in SPLITS]
        lines.append(f"| {lab.title()} | " + " | ".join(cells) + " |")
    lines += ["", "## Tokens per sentence", "",
              "| Split | mean | median | p90 | p95 | p99 | max |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split in SPLITS:
        d = stats[split]["tokens_per_sentence"]
        lines.append(f"| {split} | {d['mean']} | {d['median']:.0f} | {d['p90']:.0f} "
                     f"| {d['p95']:.0f} | {d['p99']:.0f} | {d['max']} |")
    lines += ["", "## Sentences per abstract", "",
              "| Split | mean | median | p95 | max |", "|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        d = stats[split]["sentences_per_abstract"]
        lines.append(f"| {split} | {d['mean']} | {d['median']:.0f} | {d['p95']:.0f} | {d['max']} |")
    v = stats["vocabulary"]
    lines += ["", "## Vocabulary (train, lowercased whitespace tokens)", "",
              f"- Distinct tokens: {v['size']:,}",
              f"- Tokens seen at least twice: {v['min_count_2']:,}",
              f"- Validation tokens not in train vocabulary (OOV rate): {v['validation_oov_percent']}%"]
    if "max_len_coverage" in stats:
        c = stats["max_len_coverage"]
        lines += ["", f"## Coverage of max_len = {c['max_len']}", "",
                  f"- Train sentences with at most {c['max_len']} tokens: {c['train_percent']}%"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=Path("data/hf_cache"),
                    help="where Hugging Face stores the downloaded dataset")
    ap.add_argument("--out-dir", type=Path, default=Path("eda_output"))
    ap.add_argument("--max-len", type=int, default=None,
                    help="report the share of train sentences that fit in this many tokens")
    args = ap.parse_args()

    data = load_records(args.cache_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"dataset": f"PubMed 20k RCT (Hugging Face: {HF_DATASET})"}
    for split, records in data.items():
        stats[split] = split_stats(records)

    train_vocab = Counter(tok.lower() for r in data["train"] for tok in r["text"].split())
    val_tokens = [tok.lower() for r in data["validation"] for tok in r["text"].split()]
    stats["vocabulary"] = {
        "size": len(train_vocab),
        "min_count_2": sum(1 for c in train_vocab.values() if c >= 2),
        "validation_oov_percent": round(100 * sum(t not in train_vocab for t in val_tokens) / len(val_tokens), 2),
    }

    train_lengths = [r["n_tokens"] for r in data["train"]]
    if args.max_len:
        stats["max_len_coverage"] = {
            "max_len": args.max_len,
            "train_percent": round(100 * float(np.mean(np.asarray(train_lengths) <= args.max_len)), 2),
        }

    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_summary(stats, args.out_dir / "summary.md")

    plot_class_distribution(stats, args.out_dir / "class_distribution.png")
    plot_length_hist(train_lengths, stats["train"]["tokens_per_sentence"], args.max_len,
                     args.out_dir / "sentence_length_hist.png")
    plot_sentences_per_abstract(data["train"], args.out_dir / "sentences_per_abstract.png")
    plot_label_by_position(data["train"], args.out_dir / "label_by_position.png")
    plot_length_by_class(data["train"], args.out_dir / "length_by_class.png")

    print((args.out_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"Figures and stats written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
