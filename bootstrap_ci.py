"""Bootstrap 95% confidence intervals for macro-F1 on a shared set of test sentences.

Resamples whole abstracts (not sentences), because sentences inside one abstract
are correlated. Every run must have predictions for exactly the same sentences,
which is the case for the LLM sample and the *_on_llm_sample re-scores.

    python bootstrap_ci.py                     # default: CNN, BiLSTM, MLP, BiomedBERT and the gpt-6.2-sol runs
    python bootstrap_ci.py --runs "CNN=cnn_final_on_llm_sample" "BiLSTM=lstm_final_on_llm_sample" \
                           --reference BiLSTM --n-boot 2000

Writes results/bootstrap_ci.md (tables for the report) and results/bootstrap_ci.json.
"""

import argparse
import json

import numpy as np
from sklearn.metrics import f1_score

from rct.data import load_splits
from rct.evaluate import RESULTS_DIR, read_prediction_file

DEFAULT_RUNS = [
    "CNN=cnn_final_on_llm_sample",
    "BiLSTM=lstm_final_on_llm_sample",
    "MLP=mlp_final_on_llm_sample",
    "BiomedBERT=bert_final_on_llm_sample",
    "gpt-6.2-sol sentence 0-shot=llm_gpt-6.2-sol_sentence_0shot",
    "gpt-6.2-sol abstract 0-shot=llm_gpt-6.2-sol_abstract_0shot",
    "gpt-6.2-sol abstract 3-shot=llm_gpt-6.2-sol_abstract_3shot",
]


def macro_f1(y, p):
    return 100 * f1_score(y, p, average="macro", labels=range(5), zero_division=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help='"Label=results_dir" pairs')
    ap.add_argument("--reference", default="BiLSTM", help="label every other run is compared against")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bootstrap_ci", help="file stem under results/")
    args = ap.parse_args(argv)

    runs = dict(r.split("=", 1) for r in args.runs)
    if args.reference not in runs:
        raise SystemExit(f"--reference {args.reference!r} is not one of {list(runs)}")

    preds, idx = {}, None
    for label, name in runs.items():
        i, p = read_prediction_file(RESULTS_DIR / name / f"{args.split}_predictions.csv")
        order = np.argsort(i)
        i, p = i[order], p[order]
        if idx is None:
            idx = i
        elif not np.array_equal(idx, i):
            raise SystemExit(f"{name} covers different sentences than {next(iter(runs.values()))}")
        preds[label] = p

    split = load_splits()[args.split]
    y = split.labels[idx]
    abstracts = np.array([split.abstract_ids[i] for i in idx])
    groups = [np.flatnonzero(abstracts == a) for a in dict.fromkeys(abstracts)]

    rng = np.random.default_rng(args.seed)
    boot = {k: np.empty(args.n_boot) for k in preds}
    for b in range(args.n_boot):
        s = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        for k, p in preds.items():
            boot[k][b] = macro_f1(y[s], p[s])

    ref = args.reference
    rows = []
    for k, p in preds.items():
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        row = {"run": k, "dir": runs[k], "macro_f1": round(macro_f1(y, p), 2), "ci95": [round(lo, 1), round(hi, 1)]}
        if k != ref:
            d = boot[k] - boot[ref]
            dlo, dhi = np.percentile(d, [2.5, 97.5])
            row["diff_vs_ref"] = round(macro_f1(y, p) - macro_f1(y, preds[ref]), 2)
            row["diff_ci95"] = [round(dlo, 1), round(dhi, 1)]
            row["significant"] = bool(dlo > 0 or dhi < 0)
        rows.append(row)

    meta = {"split": args.split, "sentences": int(len(idx)), "abstracts": len(groups),
            "n_boot": args.n_boot, "seed": args.seed, "reference": ref}
    lines = [
        f"# Bootstrap confidence intervals ({meta['abstracts']} {args.split} abstracts, "
        f"{meta['sentences']:,} sentences, {args.n_boot:,} resamples of whole abstracts)",
        "",
        "| Run | Macro-F1 | 95% CI |",
        "|---|---:|---|",
        *[f"| {r['run']} | {r['macro_f1']:.2f} | [{r['ci95'][0]}, {r['ci95'][1]}] |" for r in rows],
        "",
        f"## Difference in macro-F1 vs {ref} (paired: same resampled abstracts for both runs)",
        "",
        "| Run | Difference | 95% CI | Significant |",
        "|---|---:|---|---|",
        *[f"| {r['run']} | {r['diff_vs_ref']:+.2f} | [{r['diff_ci95'][0]:+.1f}, {r['diff_ci95'][1]:+.1f}] | "
          f"{'yes' if r['significant'] else 'no'} |" for r in rows if r["run"] != ref],
        "",
        "Significant = the 95% interval of the difference does not contain 0.",
    ]
    md = "\n".join(lines) + "\n"
    (RESULTS_DIR / f"{args.out}.md").write_text(md, encoding="utf-8")
    (RESULTS_DIR / f"{args.out}.json").write_text(json.dumps({"meta": meta, "runs": rows}, indent=2))
    print(md)


if __name__ == "__main__":
    main()
