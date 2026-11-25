"""Hyperparameter search for the TF-IDF MLP over the proposal's search space (slide 10).

Like sweep_nn.py, this changes one hyperparameter at a time around the start
config (validation only, never test). Pick the best values from the summary
table, then do one final run with --eval-test.

    python sweep_mlp.py
    python sweep_mlp.py --only start,d0.3
    python sweep_mlp.py --dry-run          # just list the runs
    python -m rct.evaluate summary

Search space: hidden sizes {256, 512, 1024} (first layer; later layers halve),
depth {1, 2, 3}, dropout {0.3, 0.5}, TF-IDF features {20k, 50k}, Adam lr {1e-3, 3e-4}.
"""

import argparse

import train_mlp

# (name suffix, extra CLI args) per variation; the start config itself comes first.
SPACE = [
    ("start", []),                                # 512 -> 256, dropout 0.5, 50k, lr 1e-3
    ("h256", ["--hidden", "256,128"]),
    ("h1024", ["--hidden", "1024,512"]),
    ("depth1", ["--hidden", "512"]),
    ("depth3", ["--hidden", "512,256,128"]),
    ("d0.3", ["--dropout", "0.3"]),
    ("tf20k", ["--max-features", "20000"]),
    ("lr3e-4", ["--lr", "3e-4"]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated suffixes to run, e.g. start,d0.3")
    ap.add_argument("--prefix", default="sweep_mlp", help="run names are <prefix>_<suffix>")
    ap.add_argument("--dry-run", action="store_true")
    args, passthrough = ap.parse_known_args()  # anything else (e.g. --epochs 5) goes to every run

    runs = SPACE
    if args.only:
        keep = set(args.only.split(","))
        runs = [r for r in runs if r[0] in keep]

    for suffix, extra in runs:
        argv = ["--name", f"{args.prefix}_{suffix}", *extra, *passthrough]
        print("python train_mlp.py " + " ".join(argv), flush=True)
        if not args.dry_run:
            train_mlp.run(argv)


if __name__ == "__main__":
    main()
