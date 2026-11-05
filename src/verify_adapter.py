"""Проверка загрузки пробного адаптера отдельным процессом на T4."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import ROOT, config, read_jsonl, write_json
from .modeling import generate_one, load_model, resolve_revisions


def verify(run_dir):
    cfg = config()
    adapter = run_dir / "smoke_adapter"
    if not (adapter / "adapter_model.safetensors").exists():
        raise RuntimeError("Пробный адаптер не найден")
    model, tokenizer = load_model(cfg["model_id"], resolve_revisions(run_dir)[cfg["model_id"]], adapter)
    row = read_jsonl(ROOT / "data/validation.jsonl")[0]
    result = generate_one(model, tokenizer, row["messages"][:-1], 32)
    if not result["output_tokens"]:
        raise RuntimeError("После загрузки адаптера нет токенов")
    write_json(run_dir / "smoke_adapter_reload.json", {"status": "passed", "scope": "Проверка загрузки и генерации, не оценка качества", **result})
    print("Пробный адаптер загружен, генерация выполнена", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    verify(parser.parse_args().run_dir)
