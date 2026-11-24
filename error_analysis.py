"""Error analysis across models from saved test predictions (no training, no API calls).

    python error_analysis.py
    python error_analysis.py --full "CNN=cnn_final" "BiLSTM=lstm_final" "MLP=mlp_final" \
                             --sample "CNN=cnn_final_on_llm_sample" "BiLSTM=lstm_final_on_llm_sample" \
                                      "GPT 3-shot=llm_gpt-6.2-sol_abstract_3shot"

--full    runs scored on the whole test set (error types, Background/Objective, position)
--sample  runs scored on the same 200-abstract LLM sample (error types, agreement, examples)

Writes results/error_analysis/report.md and error_by_position.png.
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rct.data import LABELS, load_splits
from rct.evaluate import RESULTS_DIR, read_prediction_file

OUT_DIR = RESULTS_DIR / "error_analysis"
SHORT = {l: l.capitalize() for l in LABELS}
BG, OBJ = LABELS.index("BACKGROUND"), LABELS.index("OBJECTIVE")

# Palette shared with the proposal slides; each series also gets its own marker and dash.
INK, MUTED, GRID = "#14213D", "#5B6573", "#E3E7ED"
SERIES_STYLE = [("#1F5FA8", "o", "-"), ("#D9822B", "s", "--"), ("#5B6573", "^", ":"), ("#14213D", "D", "-.")]

DEFAULT_FULL = ["CNN=cnn_final", "BiLSTM=lstm_final"]
DEFAULT_SAMPLE = [
    "CNN=cnn_final_on_llm_sample",
    "BiLSTM=lstm_final_on_llm_sample",
    "GPT sentence 0-shot=llm_gpt-6.2-sol_sentence_0shot",
    "GPT abstract 3-shot=llm_gpt-6.2-sol_abstract_3shot",
]


def load_runs(pairs, test):
    runs = {}
    for pair in pairs:
        label, name = pair.split("=", 1)
        idx, pred = read_prediction_file(RESULTS_DIR / name / "test_predictions.csv")
        order = np.argsort(idx)
        runs[label] = (idx[order], pred[order])
    return runs


def md_table(header, rows, left=1):
    """First `left` columns left-aligned, the rest right-aligned (numbers)."""
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i < left else "---:" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def error_types(runs, test, top=6):
    lines = []
    for label, (idx, pred) in runs.items():
        y = test.labels[idx]
        wrong = y != pred
        pairs = Counter((LABELS[a], LABELS[b]) for a, b in zip(y[wrong], pred[wrong]))
        total = int(wrong.sum())
        rows = [[f"{SHORT[a]} → {SHORT[b]}", c, f"{100 * c / total:.0f}%"] for (a, b), c in pairs.most_common(top)]
        bo = pairs[("BACKGROUND", "OBJECTIVE")] + pairs[("OBJECTIVE", "BACKGROUND")]
        lines += [f"**{label}**: {total:,} errors in {len(idx):,} sentences ({100 * total / len(idx):.1f}%); "
                  f"Background ↔ Objective = {bo:,} ({100 * bo / total:.0f}% of errors)", "",
                  md_table(["True → predicted", "Errors", "Share"], rows), ""]
    return lines


def abstract_label_sets(test):
    sets = {}
    for a, y in zip(test.abstract_ids, test.labels):
        sets.setdefault(a, set()).add(int(y))
    return sets


def background_objective(runs, test):
    """Are Background/Objective errors about abstracts that use only one of the two labels?"""
    sets = abstract_label_sets(test)
    first = test.labels[test.sentence_ids == 0]
    n_abs = len(sets)
    has_bg = sum(BG in s for s in sets.values())
    has_obj = sum(OBJ in s for s in sets.values())
    both = sum(BG in s and OBJ in s for s in sets.values())
    lines = [
        f"In the {n_abs:,} test abstracts, {100 * has_bg / n_abs:.0f}% contain a Background sentence, "
        f"{100 * has_obj / n_abs:.0f}% an Objective sentence and only {100 * both / n_abs:.0f}% both. "
        f"The first sentence is Background in {100 * np.mean(first == BG):.0f}% of abstracts and "
        f"Objective in {100 * np.mean(first == OBJ):.0f}%. Which label the opening sentences get depends "
        "largely on whether the authors used a 'Background' or an 'Objective' heading.",
        "",
    ]
    rows = []
    for label, (idx, pred) in runs.items():
        y = test.labels[idx]
        abs_ids = [test.abstract_ids[i] for i in idx]
        o2b = (y == OBJ) & (pred == BG)
        b2o = (y == BG) & (pred == OBJ)
        o2b_no_bg = sum(BG not in sets[abs_ids[k]] for k in np.flatnonzero(o2b))
        b2o_no_obj = sum(OBJ not in sets[abs_ids[k]] for k in np.flatnonzero(b2o))
        rows.append([label, int(o2b.sum()), f"{100 * o2b_no_bg / max(1, o2b.sum()):.0f}%",
                     int(b2o.sum()), f"{100 * b2o_no_obj / max(1, b2o.sum()):.0f}%"])
    lines += [md_table(["Model", "Objective → Background", "…in abstracts with no Background",
                        "Background → Objective", "…in abstracts with no Objective"], rows), ""]
    return lines


def merged_scores(runs, test):
    """Scores with Objective folded into Background: how much of the gap is that one pair?"""
    from sklearn.metrics import accuracy_score, f1_score

    def merge(a):
        return np.where(a == OBJ, BG, a)

    rows = []
    for label, (idx, pred) in runs.items():
        y = test.labels[idx]
        rows.append([label, f"{len(idx):,}",
                     f"{100 * accuracy_score(y, pred):.1f}", f"{100 * f1_score(y, pred, average='macro'):.1f}",
                     f"{100 * accuracy_score(merge(y), merge(pred)):.1f}",
                     f"{100 * f1_score(merge(y), merge(pred), average='macro'):.1f}"])
    return ["Same predictions, scored again after treating Objective and Background as one class "
            "(4 classes). Nothing is retrained.", "",
            md_table(["Model", "Sentences", "5-class accuracy", "5-class macro-F1",
                      "Merged accuracy", "Merged macro-F1"], rows), ""]


def position_bins(test, idx, max_pos=10):
    pos = test.sentence_ids[idx] + 1
    return np.minimum(pos, max_pos + 1)  # last bin = "11+"


def error_by_position(runs, test, out_png, max_pos=10):
    bins = list(range(1, max_pos + 2))
    names = [str(b) for b in bins[:-1]] + [f"{max_pos + 1}+"]
    rates = {}
    for label, (idx, pred) in runs.items():
        b = position_bins(test, idx, max_pos)
        wrong = test.labels[idx] != pred
        rates[label] = [100 * wrong[b == k].mean() for k in bins]

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    for (label, r), (color, marker, dash) in zip(rates.items(), SERIES_STYLE):
        ax.plot(bins, r, color=color, linestyle=dash, linewidth=2, marker=marker, markersize=6,
                markeredgecolor="white", markeredgewidth=1.5, label=label, zorder=3)
        ax.annotate(label, (bins[0], r[0]), xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=9, color=INK)
    ax.set_xticks(bins, names)
    ax.set_xlim(0.2, bins[-1] + 0.4)
    ax.set_ylim(0, max(max(r) for r in rates.values()) * 1.15)
    ax.set_xlabel("Sentence position in the abstract", color=MUTED)
    ax.set_ylabel("Error rate (%)", color=MUTED)
    ax.set_title("Error rate is highest in the first two sentences (test set)", loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    rows = [[n] + [f"{rates[l][k]:.1f}" for l in rates] for k, n in enumerate(names)]
    counts = Counter(position_bins(test, next(iter(runs.values()))[0], max_pos))
    rows = [r + [f"{counts[bins[k]]:,}"] for k, r in enumerate(rows)]
    return [md_table(["Position"] + [f"{l} error %" for l in rates] + ["Sentences"], rows), ""]


def agreement(runs, test, n_examples, seed):
    labels = list(runs)
    idx0 = runs[labels[0]][0]
    for l in labels[1:]:
        if not np.array_equal(runs[l][0], idx0):
            raise SystemExit(f"--sample runs must cover the same sentences ({l} differs)")
    y = test.labels[idx0]
    wrong = np.stack([runs[l][1] != y for l in labels])
    n_wrong = wrong.sum(0)
    lines = [md_table(["Models wrong", "Sentences", "Share"],
                      [[k, int((n_wrong == k).sum()), f"{100 * np.mean(n_wrong == k):.1f}%"]
                       for k in range(len(labels) + 1)]), ""]
    all_wrong = n_wrong == len(labels)
    by_true = Counter(LABELS[t] for t in y[all_wrong])
    lines += [f"Sentences every model gets wrong, by true label: "
              + ", ".join(f"{SHORT[k]} {v}" for k, v in by_true.most_common()), ""]

    preds = np.stack([runs[l][1] for l in labels])
    same = (preds == preds[0]).all(0) & all_wrong
    lines += [f"Of these {int(all_wrong.sum())}, {int(same.sum())} get the same wrong label from every model, "
              "a hint that the gold label rather than the models may be the odd one out.", ""]

    rng = np.random.default_rng(seed)
    pick = np.flatnonzero(all_wrong)
    pick = np.sort(rng.choice(pick, min(n_examples, len(pick)), replace=False)) if len(pick) else pick
    rows = []
    for k in pick:
        i = idx0[k]
        text = test.texts[i] if len(test.texts[i]) <= 220 else test.texts[i][:217] + "..."
        rows.append([f"{test.sentence_ids[i] + 1}/{test.total_lines[i]}", SHORT[LABELS[y[k]]],
                     " / ".join(SHORT[LABELS[preds[m, k]]] for m in range(len(labels))),
                     text.replace("|", "\\|")])
    lines += [f"Random examples (predictions in the order {', '.join(labels)}):", "",
              md_table(["Pos", "Gold", "Predicted", "Sentence"], rows, left=4), ""]
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", nargs="+", default=DEFAULT_FULL, help='"Label=results_dir" on the whole test set')
    ap.add_argument("--sample", nargs="+", default=DEFAULT_SAMPLE, help='"Label=results_dir" on the LLM sample')
    ap.add_argument("--examples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    test = load_splits()["test"]
    full = load_runs(args.full, test)
    sample = load_runs(args.sample, test)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = ["# Error analysis", "",
             f"Full test set ({len(test):,} sentences): {', '.join(full)}. "
             f"Shared 200-abstract sample ({len(next(iter(sample.values()))[0]):,} sentences): {', '.join(sample)}.", "",
             "## 1. What kind of errors", "", "### Full test set", ""]
    lines += error_types(full, test)
    lines += ["### LLM sample", ""]
    lines += error_types({k: v for k, v in sample.items() if k not in full}, test)
    lines += ["## 2. Background vs Objective", ""]
    lines += background_objective({**full, **{k: v for k, v in sample.items() if k not in full}}, test)
    lines += ["## 3. Where in the abstract", "", "![Error rate by position](error_by_position.png)", ""]
    lines += error_by_position(full, test, OUT_DIR / "error_by_position.png")
    lines += ["## 4. Do the models fail on the same sentences? (LLM sample)", ""]
    lines += agreement(sample, test, args.examples, args.seed)
    lines += ["## 5. Scores with Background and Objective merged", "", "### Full test set", ""]
    lines += merged_scores(full, test)
    lines += ["### LLM sample", ""]
    lines += merged_scores(sample, test)[2:]

    md = "\n".join(lines)
    (OUT_DIR / "report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
