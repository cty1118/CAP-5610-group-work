"""Compare current LLMs (OpenAI API) with our models on a sample of the test set.

The key is read from .env (OPENAI_API_KEY=...), which git ignores. It is never printed.

Settings
    sentence   each sentence alone plus its position ("sentence 3 of 11"): the same
               input our MLP / CNN / LSTM / BERT get. Sentences from many abstracts are
               shuffled and sent in batches, so no request sees an abstract's context.
    abstract   one request per abstract with every sentence numbered: the model sees
               the context, like the paper's CRF / bi-ANN.
    --shots K  few-shot: K labelled training abstracts go into the instructions.

Usage
    python llm_compare.py --model gpt-6.2-sol --setting abstract --max-requests 5   # trial + cost estimate
    python llm_compare.py --model gpt-6.2-sol --setting abstract                    # full run, then scored
    python llm_compare.py --baselines cnn_final,lstm_final                          # our models on the same sample

Every response is cached in results/<run>/responses.jsonl, so an interrupted run
resumes where it stopped and nothing is paid for twice.
"""

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from rct.data import LABELS, load_splits
from rct.evaluate import RESULTS_DIR, read_prediction_file, save_run

ROOT = Path(__file__).resolve().parent
SAMPLE_FILE = RESULTS_DIR / "llm_sample_test_abstracts.txt"
FALLBACK = "METHODS"  # used only if a sentence still has no valid label after retries

LABEL_GUIDE = """Labels:
BACKGROUND: context, prior knowledge, or the problem that motivates the study.
OBJECTIVE: the aim, purpose, or hypothesis of this particular study.
METHODS: design, setting, participants, interventions, measurements, or analysis.
RESULTS: findings of the study, including numbers and statistical outcomes.
CONCLUSIONS: interpretation, implications, or recommendations drawn from the findings."""

TASK = {
    "abstract": (
        "You label every sentence of a randomized controlled trial abstract with its role. "
        "Sentences are numbered [1], [2], ... Numbers in the text were replaced by '@'. "
        "Return exactly one label for every sentence number."
    ),
    "sentence": (
        "You label sentences from randomized controlled trial abstracts with their role. "
        "Each numbered item is a single sentence from a different abstract, shown alone with its "
        "position in its own abstract. Judge every item independently. Numbers in the text were "
        "replaced by '@'. Return exactly one label for every item number."
    ),
}

SCHEMA = {
    "type": "json_schema",
    "name": "sentence_labels",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["labels"],
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "label"],
                    "properties": {"id": {"type": "integer"}, "label": {"type": "string", "enum": LABELS}},
                },
            }
        },
    },
}


def load_env(path=ROOT / ".env"):
    if path.exists():
        for line in path.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() and not k.strip().startswith("#"):
                os.environ.setdefault(k.strip(), v.strip())


def group_by_abstract(split):
    groups = {}
    for i, a in enumerate(split.abstract_ids):
        groups.setdefault(a, []).append(i)
    return groups  # insertion order = order in the split; rows are already in sentence order


def sample_abstracts(test, n, seed):
    """Fixed random sample of test abstracts, saved once so every run uses the same one."""
    if SAMPLE_FILE.exists():
        ids = SAMPLE_FILE.read_text().split()
        if len(ids) != n:
            raise SystemExit(f"{SAMPLE_FILE} holds {len(ids)} abstracts, not {n}; delete it to resample")
        return ids
    all_ids = list(group_by_abstract(test))
    ids = [all_ids[i] for i in sorted(np.random.default_rng(seed).choice(len(all_ids), n, replace=False))]
    SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_FILE.write_text("\n".join(ids) + "\n")
    return ids


def item_text(split, i, with_position):
    text = split.texts[i]
    if with_position:
        return f"(sentence {split.sentence_ids[i] + 1} of {split.total_lines[i]}) {text}"
    return text


def build_instructions(setting, train, shots, seed):
    parts = [TASK[setting], LABEL_GUIDE]
    if shots:
        groups = group_by_abstract(train)
        ids = list(groups)
        pick = np.random.default_rng(seed).choice(len(ids), shots, replace=False)
        parts.append("Labelled examples from other abstracts:")
        for k, j in enumerate(pick, 1):
            rows = groups[ids[j]]
            if setting == "abstract":
                lines = [f"[{n}] {train.texts[i]} -> {LABELS[train.labels[i]]}" for n, i in enumerate(rows, 1)]
            else:
                lines = [f"{item_text(train, i, True)} -> {LABELS[train.labels[i]]}" for i in rows]
            parts.append(f"Example {k}:\n" + "\n".join(lines))
    return "\n\n".join(parts)


def build_requests(setting, test, sample_ids, batch_size, seed):
    """List of {key, rows}; rows are test row indices, item n of a request is rows[n-1]."""
    groups = group_by_abstract(test)
    if setting == "abstract":
        return [{"key": a, "rows": groups[a]} for a in sample_ids]
    rows = [i for a in sample_ids for i in groups[a]]
    rows = [rows[j] for j in np.random.default_rng(seed).permutation(len(rows))]
    return [{"key": f"batch{b:04d}", "rows": rows[s:s + batch_size]}
            for b, s in enumerate(range(0, len(rows), batch_size))]


def request_input(setting, test, rows):
    pos = setting == "sentence"
    return "\n".join(f"[{n}] {item_text(test, i, pos)}" for n, i in enumerate(rows, 1))


def call_model(client, args, instructions, text, n_items):
    """One API call with retries until every item has a valid label. Returns (labels, usage)."""
    kwargs = {"model": args.model, "instructions": instructions, "input": text, "text": {"format": SCHEMA}}
    if args.reasoning_effort:
        kwargs["reasoning"] = {"effort": args.reasoning_effort}
    usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    labels = {}
    for _ in range(args.retries + 1):
        r = client.responses.create(**kwargs)
        usage["calls"] += 1
        usage["input_tokens"] += r.usage.input_tokens
        usage["output_tokens"] += r.usage.output_tokens
        try:
            got = {int(x["id"]): x["label"] for x in json.loads(r.output_text)["labels"]}
        except (ValueError, KeyError, TypeError):
            continue
        labels.update({k: v for k, v in got.items() if 1 <= k <= n_items and v in LABELS})
        if len(labels) == n_items:
            break
    return labels, usage


def run_llm(args, splits):
    from openai import OpenAI

    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not found: put it in .env (see README)")
    client = OpenAI(max_retries=5)

    test = splits["test"]
    sample_ids = sample_abstracts(test, args.n_abstracts, args.seed)
    instructions = build_instructions(args.setting, splits["train"], args.shots, args.seed)
    requests = build_requests(args.setting, test, sample_ids, args.batch_size, args.seed)

    name = args.name or f"llm_{args.model}_{args.setting}_{args.shots}shot"
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "responses.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            rec = json.loads(line)
            done[rec["key"]] = rec

    todo = [r for r in requests if r["key"] not in done]
    if args.max_requests:
        todo = todo[: max(0, args.max_requests - len(done))]
    print(f"{name}: {len(requests)} requests over {sum(len(r['rows']) for r in requests):,} sentences "
          f"({len(done)} cached, {len(todo)} to send)")

    lock = threading.Lock()

    def work(req):
        text = request_input(args.setting, test, req["rows"])
        labels, usage = call_model(client, args, instructions, text, len(req["rows"]))
        rec = {"key": req["key"], "rows": req["rows"],
               "labels": {str(k): v for k, v in sorted(labels.items())}, "usage": usage}
        with lock:
            with open(cache, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(work, r) for r in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            done[rec["key"]] = rec
            if k % 10 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} requests sent")

    used = [done[r["key"]] for r in requests if r["key"] in done]
    tin = sum(r["usage"]["input_tokens"] for r in used)
    tout = sum(r["usage"]["output_tokens"] for r in used)
    print(f"tokens so far: input {tin:,}  output {tout:,}  over {len(used)} requests")
    if len(used) < len(requests):
        scale = len(requests) / max(1, len(used))
        print(f"estimate for all {len(requests)} requests: input ~{int(tin * scale):,}  output ~{int(tout * scale):,} "
              f"tokens. Multiply by the model's per-token prices (platform.openai.com pricing) for the cost.")
        print("Not scored: run again without --max-requests to finish.")
        return

    idx, pred, missing = [], [], 0
    for r in used:
        for n, i in enumerate(r["rows"], 1):
            lab = r["labels"].get(str(n))
            if lab is None:
                missing += 1
                lab = FALLBACK
            idx.append(i)
            pred.append(LABELS.index(lab))
    idx, pred = np.array(idx), np.array(pred)
    order = np.argsort(idx)
    idx, pred = idx[order], pred[order]
    config = {k: v for k, v in vars(args).items() if k not in ("baselines",)}
    extra = {"input_tokens": tin, "output_tokens": tout, "requests": len(used),
             "api_calls": sum(r["usage"]["calls"] for r in used), "missing_labels": missing,
             "fallback_label": FALLBACK, "n_abstracts": len(sample_ids)}
    if missing:
        print(f"warning: {missing} sentences had no valid label after retries; counted as {FALLBACK}")
    save_run(name, "test", test.labels[idx], pred, indices=idx, config=config, extra=extra)


def score_baselines(names, splits, n, seed):
    """Score our models' saved test predictions on the same sampled abstracts."""
    test = splits["test"]
    sample_ids = set(sample_abstracts(test, n, seed))
    keep = np.array([a in sample_ids for a in test.abstract_ids])
    for name in names:
        path = RESULTS_DIR / name / "test_predictions.csv"
        idx, pred = read_prediction_file(path)
        mask = keep[idx]
        save_run(f"{name}_on_llm_sample", "test", test.labels[idx[mask]], pred[mask], indices=idx[mask],
                 config={"source": str(path.relative_to(ROOT)), "n_abstracts": n})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="OpenAI model id, e.g. gpt-6.2-sol or gpt-6-luna")
    ap.add_argument("--setting", choices=["sentence", "abstract"], default="abstract")
    ap.add_argument("--shots", type=int, default=0, help="labelled training abstracts in the prompt")
    ap.add_argument("--n-abstracts", type=int, default=200, help="test abstracts in the shared sample")
    ap.add_argument("--batch-size", type=int, default=25, help="sentences per request (sentence setting)")
    ap.add_argument("--reasoning-effort", help="e.g. low / medium, for reasoning models (default: model's own)")
    ap.add_argument("--max-requests", type=int, default=0, help="stop after N requests in total (trial run)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=2, help="re-asks when labels are missing or invalid")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", help="run name (default llm_<model>_<setting>_<k>shot)")
    ap.add_argument("--baselines", help="comma-separated result dirs to score on the same sample")
    args = ap.parse_args(argv)

    splits = load_splits()
    if args.baselines:
        score_baselines(args.baselines.split(","), splits, args.n_abstracts, args.seed)
    if args.model:
        run_llm(args, splits)
    elif not args.baselines:
        ap.error("give --model and/or --baselines")


if __name__ == "__main__":
    main()
