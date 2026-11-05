"""Последовательный запуск, checkpoints и журнал стадий. Работает без API-ключей."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from .common import ROOT, config, digest, environment, now, read_json, sha256, write_json


def fingerprint():
    paths = []
    for directory in ["src", "configs", "prompts", "data"]:
        paths.extend(path for path in (ROOT / directory).rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    paths.append(ROOT / "requirements-colab.txt")
    return digest({str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)})


def package_results(run_dir):
    destination = run_dir / "houseplants_results.zip"
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file() or path in (destination, temporary):
                continue
            relative = path.relative_to(run_dir)
            if any(part.endswith("checkpoints") or part == "smoke_adapter" for part in relative.parts):
                continue
            if path.name in (".pipeline.lock",):
                continue
            archive.write(path, relative)
        for folder in ["data", "src", "configs", "prompts", "tests", "scripts", "report"]:
            for path in sorted((ROOT / folder).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    archive.write(path, Path("project") / path.relative_to(ROOT))
        for name in ["README.md", "PLAN.md", "MODEL_CARD.md", "requirements-colab.txt", "requirements-dev.txt"]:
            archive.write(ROOT / name, Path("project") / name)
    os.replace(temporary, destination)
    return destination


def pipeline(run_dir):
    import fcntl
    run_dir.mkdir(parents=True, exist_ok=True)
    # Drive FUSE может не поддерживать flock; блокировка нужна на локальной FS.
    lock_path = Path(tempfile.gettempdir()) / ("houseplants-" + digest(str(run_dir.resolve()))[:16] + ".lock")
    lock = lock_path.open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("В этой папке уже запущен pipeline; дождитесь завершения.") from error
    identifier = fingerprint()
    state_path = run_dir / "state.json"
    state = read_json(state_path) if state_path.exists() else {"fingerprint": identifier, "created_at": now(), "completed": []}
    if state["fingerprint"] != identifier:
        raise RuntimeError("Код, данные или конфигурация изменились. Выберите новую папку результатов, чтобы не смешать эксперименты.")
    write_json(state_path, state)
    logs = run_dir / "logs"
    logs.mkdir(exist_ok=True)
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_XET="1", PYTHONUNBUFFERED="1", WANDB_DISABLED="true")

    def stage(name, module, *arguments):
        if name in state["completed"]:
            print("Уже выполнено:", name, flush=True)
            return
        print("\nЭтап:", name, flush=True)
        started = time.monotonic()
        command = [sys.executable, "-m", module, *map(str, arguments)]
        with (logs / (name + ".log")).open("a", encoding="utf-8") as log:
            log.write(f"\n{now()} {command!r}\n")
            with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            if code:
                state["last_error"] = {"stage": name, "returncode": code, "at": now()}
                write_json(state_path, state)
                raise RuntimeError(f"Этап {name} завершился с кодом {code}. Лог: {logs / (name + '.log')}. Повторный запуск продолжит с этого этапа.")
        state["completed"].append(name)
        state.pop("last_error", None)
        state.setdefault("durations_seconds", {})[name] = time.monotonic()-started
        write_json(state_path, state)

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна: выберите в Colab T4 GPU и выполните ноутбук снова.")
        memory = torch.cuda.get_device_properties(0).total_memory/2**30
        if memory < 14:
            raise RuntimeError(f"Обнаружено {memory:.1f} GiB GPU-памяти. Целевая конфигурация рассчитана на T4 с 16 GB.")
        info = environment() | {"gpu": torch.cuda.get_device_name(0), "gpu_total_memory_gib": memory}
        write_json(run_dir / "environment.json", info)
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        (run_dir / "pip-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
        write_json(run_dir / "experiment.json", config())
        stage("01_data_audit", "src.prepare_data", "--audit-only")
        stage("02_format_check", "src.checks")
        from .modeling import resolve_revisions
        resolve_revisions(run_dir)
        stage("03_first_answers", "src.generate", "--run-dir", run_dir, "--split", "validation", "--variant", "baseline", "--limit", 2)
        stage("04_pilot_judge", "src.judge", "--run-dir", run_dir, "--split", "validation", "--limit", 2, "--no-repeats")
        stage("05_smoke_training", "src.train", "--run-dir", run_dir, "--smoke")
        stage("05b_smoke_reload", "src.verify_adapter", "--run-dir", run_dir)
        stage("06_baseline_validation", "src.generate", "--run-dir", run_dir, "--split", "validation", "--variant", "baseline")
        stage("07_baseline_test", "src.generate", "--run-dir", run_dir, "--split", "test", "--variant", "baseline")
        stage("08_training", "src.train", "--run-dir", run_dir)
        stage("09_sft_validation", "src.generate", "--run-dir", run_dir, "--split", "validation", "--variant", "sft")
        stage("10_sft_test", "src.generate", "--run-dir", run_dir, "--split", "test", "--variant", "sft")
        stage("11_judge_validation", "src.judge", "--run-dir", run_dir, "--split", "validation", "--no-repeats")
        stage("12_judge_test", "src.judge", "--run-dir", run_dir, "--split", "test")
        stage("13_evaluation", "src.evaluate", "--run-dir", run_dir)
        stage("14_report", "src.report", "--run-dir", run_dir)
        state["status"] = "computed"
        state["finished_at"] = now()
        write_json(state_path, state)
        print("Все вычислительные этапы завершены. Проверьте полноту оценок и содержательные выводы в report.md.", flush=True)
    except Exception as error:
        state["status"] = "interrupted"
        state["pipeline_error"] = {"type": type(error).__name__, "message": str(error), "at": now()}
        write_json(state_path, state)
        raise
    finally:
        # Даже после ошибки остаётся диагностический архив; checkpoint не копируется в zip.
        for name in ["data_audit.json", "check_examples.json", "local_validation.json", "tokenization_audit.json"]:
            path = ROOT / "results" / name
            if path.exists():
                shutil.copy2(path, run_dir / name)
        if (ROOT / "README.md").exists():
            print("Архив:", package_results(run_dir), flush=True)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    pipeline(parser.parse_args().run_dir.resolve())
