"""Пересчёт сохранённых результатов; без GPU и без запросов к моделям."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import read_json, read_jsonl, write_json
from src.evaluate import score_index, summarise
from src.judge import parse_score
from src.pipeline import fingerprint


def verify(results_dir, adapter_dir, skip_weights=False):
    state = read_json(results_dir / "state.json")
    assert state["status"] == "computed"
    assert len(state["completed"]) == 15
    assert state["fingerprint"] == fingerprint(), "Исходный эксперимент изменён"
    expected = read_jsonl(ROOT / "data/test.jsonl")
    outputs = {}
    revisions = read_json(results_dir / "model_revisions.json")
    cfg = read_json(results_dir / "experiment.json")
    for variant in ["baseline", "sft"]:
        rows = read_jsonl(results_dir / f"{variant}_test.jsonl")
        assert len(rows) == len(expected) == 36
        outputs[variant] = {row["id"]:row for row in rows}
        assert set(outputs[variant]) == {row["id"] for row in expected}
        for row in rows:
            assert row["revision"] == revisions[cfg["model_id"]]
            assert row["generation"] == {"do_sample":False,"max_new_tokens":384}
            assert [message["role"] for message in row["messages"]] == ["system","user"]
    for split in ["validation", "test"]:
        for row in read_jsonl(results_dir / f"judge_{split}.jsonl"):
            assert row["revision"] == revisions[cfg["judge_model_id"]]
            assert parse_score(row["raw_attempts"][-1]) == row["score"]
    recomputed = summarise(expected, outputs, score_index(results_dir / "judge_test.jsonl"), 2000)
    stored = read_json(results_dir / "evaluation/summary.json")
    assert recomputed["metrics"] == stored["metrics"]
    assert recomputed["paired_deltas"] == stored["paired_deltas"]
    proof = {"status":"passed","fingerprint":state["fingerprint"],"test_pairs":36,"metrics":"exact_match","bootstrap":"exact_match","judge_raw_json":"validated","adapter":"not_checked"}
    if not skip_weights:
        import torch
        from safetensors import safe_open
        weights = adapter_dir / "adapter_model.safetensors"
        if weights.stat().st_size < 1024 and weights.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError("Вместо весов найден Git LFS-указатель. Выполните git lfs pull или скачайте safetensors по ссылке из README.")
        expected_hash = read_json(ROOT / "results/final_validation.json")["adapter"]["sha256"]
        assert hashlib.sha256(weights.read_bytes()).hexdigest() == expected_hash
        count, tensors = 0, 0
        with safe_open(weights, framework="pt", device="cpu") as archive:
            for key in archive.keys():
                value = archive.get_tensor(key)
                assert bool(torch.isfinite(value).all()), key
                count += value.numel()
                tensors += 1
        assert count == 29933568 and tensors == 504
        proof["adapter"] = {"sha256":expected_hash,"parameters":count,"tensors":tensors,"all_finite":True}
    write_json(ROOT / "results/submission_verification.json", proof)
    print("Проверка пройдена: 36 пар, метрики и интервалы совпали; исходные JSON судьи проверены.")
    print("Адаптер:", "проверен, 504 тензора" if not skip_weights else "пропущен по --skip-weights")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/experiment")
    parser.add_argument("--adapter-dir", type=Path, default=ROOT / "artifacts/adapter")
    parser.add_argument("--skip-weights", action="store_true")
    args = parser.parse_args()
    verify(args.results_dir, args.adapter_dir, args.skip_weights)
