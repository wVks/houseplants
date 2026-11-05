from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .common import ROOT, append_jsonl, config, digest, now, read_jsonl, write_json
from .modeling import generate_one, load_model, resolve_revisions

CRITERIA = ("correctness", "usefulness", "context", "consistency")


def parse_score(text):
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        if any(type(value.get(key)) is not int or not 1 <= value[key] <= 5 for key in CRITERIA):
            continue
        if type(value.get("critical_error")) is not bool:
            continue
        if not isinstance(value.get("reason"), str) or not value["reason"].strip():
            continue
        return {key: value[key] for key in (*CRITERIA, "critical_error", "reason")}
    raise ValueError("Судья не вернул JSON по рубрике")


def judge(run_dir, split="test", limit=None, repeats=True):
    cfg = config()
    system = (ROOT / "prompts/judge.txt").read_text().strip()
    references = {row["id"]: row for row in read_jsonl(ROOT / f"data/{split}.jsonl")}
    jobs = []
    for variant in ["baseline", "sft"]:
        answers = read_jsonl(run_dir / f"{variant}_{split}.jsonl")
        latest = {row["id"]: row for row in answers}
        for row in latest.values():
            if row.get("error") or not row.get("answer"):
                continue
            ref = references[row["id"]]
            jobs.append({"job_id": variant + ":" + row["id"], "variant": variant, "id": row["id"], "repeat": False, "payload": {"question": ref["question"], "reference_facts": ref["reference_facts"], "critical_mistakes": ref["critical_mistakes"], "answer": row["answer"]}})
    random.Random(cfg["seed"]).shuffle(jobs)
    if limit:
        jobs = jobs[:limit]
    if repeats and not limit:
        # При greedy повтор того же запроса почти всегда тождественен.
        # Меняем положение полей контекста, сохраняя факты и ответ.
        repeat_jobs = [{**job, "job_id": job["job_id"] + ":repeat", "repeat": True, "repeat_type": "reversed_payload_fields", "payload": dict(reversed(list(job["payload"].items())))} for job in reversed(jobs[:cfg["judge_repeats"]])]
        jobs += repeat_jobs
    write_json(run_dir / f"judge_{split}_order.json", [{key: value for key, value in job.items() if key != "payload"} for job in jobs])
    output = run_dir / f"judge_{split}.jsonl"
    done = {row["job_id"] for row in read_jsonl(output) if row.get("score")}
    pending = [job for job in jobs if job["job_id"] not in done]
    if not pending:
        return
    revisions = resolve_revisions(run_dir)
    model, tokenizer = load_model(cfg["judge_model_id"], revisions[cfg["judge_model_id"]])
    for number, job in enumerate(pending):
        prompt = json.dumps(job["payload"], ensure_ascii=False)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        score, errors, raw_attempts = None, [], []
        for attempt in range(2):
            current = messages if attempt == 0 else messages + [{"role": "assistant", "content": raw_attempts[-1]}, {"role": "user", "content": "Верни исправленный JSON с четырьмя целыми баллами 1–5, critical_error true/false и непустым reason. Только JSON."}]
            answer = generate_one(model, tokenizer, current, cfg["judge_max_new_tokens"])
            raw_attempts.append(answer["answer"])
            try:
                score = parse_score(answer["answer"])
                break
            except ValueError as error:
                errors.append(str(error))
        append_jsonl(output, {key: value for key, value in job.items() if key != "payload"} | {"score": score, "raw_attempts": raw_attempts, "errors": errors, "judge_model_id": cfg["judge_model_id"], "revision": revisions[cfg["judge_model_id"]], "prompt": messages, "prompt_hash": digest(messages), "at": now()})
        print(f"judge/{split}: {number+1}/{len(pending)}; valid={score is not None}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-repeats", action="store_true")
    args = parser.parse_args()
    judge(args.run_dir, args.split, args.limit, not args.no_repeats)
