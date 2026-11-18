"""Hyperparameter search for TextCNN / BiLSTM over the proposal's search space.

A full grid would be 72 CNN runs, so this changes one hyperparameter at a time
around the proposal's start config (validation only, never test). Pick the best
values from the summary table, then do one final run with --eval-test.

    python sweep_nn.py --model cnn
    python sweep_nn.py --model lstm
    python sweep_nn.py --model cnn --glove-path glove.6B.300d.txt   # adds the GloVe run
    python sweep_nn.py --model cnn --dry-run                         # just list the runs
    python -m rct.evaluate summary
"""

import argparse

import train_nn

# (name suffix, extra CLI args) per variation; the start config itself comes first.
SPACE = {
    "cnn": [
        ("start", []),
        ("w234", ["--filter-widths", "2,3,4"]),
        ("w357", ["--filter-widths", "3,5,7"]),
        ("f64", ["--n-filters", "64"]),
        ("f128", ["--n-filters", "128"]),
        ("f256", ["--n-filters", "256"]),
        ("d0.3", ["--dropout", "0.3"]),
        ("lr3e-4", ["--lr", "3e-4"]),
    ],
    "lstm": [
        ("start", []),
        ("h64", ["--hidden", "64"]),
        ("h256", ["--hidden", "256"]),
        ("l2", ["--layers", "2"]),
        ("att", ["--pooling", "attention"]),
        ("d0.2", ["--dropout", "0.2"]),
        ("d0.5", ["--dropout", "0.5"]),
        ("lr3e-4", ["--lr", "3e-4"]),
    ],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(SPACE))
    ap.add_argument("--glove-path", help="if given, also run the start config with GloVe 300-d")
    ap.add_argument("--only", help="comma-separated suffixes to run, e.g. start,att")
    ap.add_argument("--dry-run", action="store_true")
    args, passthrough = ap.parse_known_args()  # anything else (e.g. --epochs 5) goes to every run

    runs = list(SPACE[args.model])
    if args.glove_path:
        runs.append(("glove", ["--embedding", "glove", "--glove-path", args.glove_path]))
    if args.only:
        keep = set(args.only.split(","))
        runs = [r for r in runs if r[0] in keep]

    for suffix, extra in runs:
        argv = ["--model", args.model, "--name", f"sweep_{args.model}_{suffix}", *extra, *passthrough]
        print("python train_nn.py " + " ".join(argv))
        if not args.dry_run:
            train_nn.run(argv)


if __name__ == "__main__":
    main()
