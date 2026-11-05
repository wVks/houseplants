"""Подготовка кэша Hugging Face без изменения модели или параметров эксперимента."""
from __future__ import annotations

import json
import time
from pathlib import Path


def is_transient_download_error(error):
    from requests.exceptions import ChunkedEncodingError, ConnectionError, HTTPError, Timeout
    if isinstance(error, (ChunkedEncodingError, ConnectionError, Timeout)):
        return True
    if isinstance(error, HTTPError) and error.response is not None:
        return error.response.status_code == 429 or 500 <= error.response.status_code < 600
    return False


def download_snapshot(model_id, revision, *, download=None, pause=time.sleep, attempts=4):
    if download is None:
        from huggingface_hub import snapshot_download
        download = snapshot_download
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            return download(
                repo_id=model_id,
                revision=revision,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
                max_workers=1,
            )
        except Exception as error:
            if not is_transient_download_error(error) or attempt + 1 == attempts:
                raise
            seconds = min(5 * 2**attempt, 30)
            print(
                f"Обрыв загрузки {model_id}: {type(error).__name__}. "
                f"Через {seconds} с — попытка {attempt + 2}/{attempts}; "
                "скачанные файлы и незавершённые части сохраняются в кэше.",
                flush=True,
            )
            pause(seconds)


def prefetch_models(run_dir, project):
    """Ревизии берутся из того же файла, который использует pipeline."""
    run_dir, project = Path(run_dir), Path(project)
    from src.modeling import resolve_revisions
    revisions = resolve_revisions(run_dir)
    cfg = json.loads((project / "configs/experiment.json").read_text())
    for model_id in dict.fromkeys([cfg["model_id"], cfg["judge_model_id"]]):
        print("Подготовка файлов модели:", model_id, flush=True)
        download_snapshot(model_id, revisions[model_id])
    print("Файлы моделей в кэше. Pipeline продолжит сохранённые этапы.", flush=True)
