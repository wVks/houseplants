from __future__ import annotations

import argparse
import time
from pathlib import Path

from .common import ROOT, append_jsonl, config, now, read_jsonl
from .modeling import generate_one, load_model, resolve_revisions


def generate(run_dir, split, variant, limit=None):
    cfg = config()
    revision = resolve_revisions(run_dir)[cfg["model_id"]]
    rows = read_jsonl(ROOT / f"data/{split}.jsonl")
    if limit:
        rows = rows[:limit]
    path = run_dir / f"{variant}_{split}.jsonl"
    done = {row["id"] for row in read_jsonl(path) if not row.get("error")}
    pending = [row for row in rows if row["id"] not in done]
    if not pending:
        return
    adapter = run_dir / "adapter" if variant == "sft" else None
    model, tokenizer = load_model(cfg["model_id"], revision, adapter=adapter)
    for index, row in enumerate(pending):
        started = time.monotonic()
        result = generate_one(model, tokenizer, row["messages"][:-1], cfg["max_new_tokens"])
        append_jsonl(path, {"id": row["id"], "group_id": row["group_id"], "variant": variant, "split": split, "question": row["question"], "messages": row["messages"][:-1], "model_id": cfg["model_id"], "revision": revision, "at": now(), "generation": {"do_sample": False, "max_new_tokens": cfg["max_new_tokens"]}, "seconds": time.monotonic()-started, **result})
        print(f"{variant}/{split}: {index+1}/{len(pending)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--variant", choices=["baseline", "sft"], required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    generate(args.run_dir, args.split, args.variant, args.limit)
