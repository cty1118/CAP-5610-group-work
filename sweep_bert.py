"""Hyperparameter search for the fine-tuned Transformer (proposal slide 11).

Fine-tuning is the most expensive step, so the search runs on a 30k-sentence
training subset (scored on the full validation set), one setting at a time
around the start config. Then the chosen config is trained on all the data.

    python sweep_bert.py --only lr2e-5,lr3e-5,lr5e-5     # step 1: learning rate
    python sweep_bert.py --only e2,len128 --lr 3e-5      # step 2: around the best lr
    python sweep_bert.py --dry-run
    python -m rct.evaluate summary

Search space: lr {2e-5, 3e-5, 5e-5}, epochs {2, 3}, max_len {64, 128};
batch 32, 10% warmup and weight decay 0.01 stay fixed.
"""

import argparse

import train_bert

# (name suffix, extra CLI args); every run uses --subset (default 30,000).
SPACE = [
    ("lr2e-5", ["--lr", "2e-5"]),               # start config: 3 epochs, max_len 64
    ("lr3e-5", ["--lr", "3e-5"]),
    ("lr5e-5", ["--lr", "5e-5"]),
    ("e2", ["--epochs", "2"]),
    ("len128", ["--max-len", "128"]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", type=int, default=30_000, help="training sentences per sweep run")
    ap.add_argument("--only", help="comma-separated suffixes to run, e.g. lr2e-5,lr3e-5")
    ap.add_argument("--prefix", default="sweep_bert", help="run names are <prefix>_<suffix>")
    ap.add_argument("--dry-run", action="store_true")
    args, passthrough = ap.parse_known_args()  # anything else (e.g. --model distilbert) goes to every run

    runs = SPACE
    if args.only:
        keep = set(args.only.split(","))
        runs = [r for r in runs if r[0] in keep]

    for suffix, extra in runs:
        argv = ["--name", f"{args.prefix}_{suffix}", "--subset", str(args.subset), *extra, *passthrough]
        print("python train_bert.py " + " ".join(argv), flush=True)
        if not args.dry_run:
            train_bert.run(argv)


if __name__ == "__main__":
    main()
