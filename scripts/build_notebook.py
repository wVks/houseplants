"""Собирает короткий Colab-ноутбук и чистую папку для загрузки в Drive."""
from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_files():
    paths = []
    for folder in ["src", "configs", "prompts", "data", "tests", "scripts", "report", "results"]:
        paths += [path for path in (ROOT / folder).rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"]
    paths += [ROOT / name for name in ["README.md", "PLAN.md", "requirements-colab.txt", "requirements-dev.txt", ".gitignore", ".gitattributes", "MODEL_CARD.md", "data/README.md"] if (ROOT / name).exists()]
    return sorted(set(paths))


def cell(kind, text):
    value = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        value.update({"execution_count": None, "outputs": []})
    return value


def build():
    cells = [cell("markdown", """# Домашние растения: SFT, оценка и отчёт

**Подготовка:** загрузите готовую папку `dist/houseplants` с компьютера в корень **Моего диска**. В ней уже есть код, данные и этот ноутбук. Ожидаемый путь: `Мой диск/houseplants/src/pipeline.py`.

Откройте `colab_pipeline.ipynb` из загруженной папки через Colab, выберите **Runtime → Change runtime type → T4 GPU**, затем **Runtime → Run all**. Подтвердите подключение Google Drive. При другом расположении папки измените только `PROJECT_DIR` в первой ячейке. GitHub и API-ключи не нужны.

Процесс выполнит первый baseline, проверку формата, пробу отдельного судьи, короткое обучение, основное SFT, сравнение до/после и отчёт. Модели скачиваются из Hugging Face, это более 20 GB. Ресурсы Colab зависят от доступности. Результаты и checkpoints сохраняются в Drive. При прерывании повторите **Run all**: завершённые этапы будут пропущены.

Данные синтетические: 288 train (96 ситуаций × 3 формулировки), 12 validation, 36 test. Эксперимент уже выполнен на T4: формат 86,11% → 100%, адаптер сохранён, итоговый отчёт включён в папку report. Содержательные ошибки и ограничения судьи разобраны в отчёте. Этот ноутбук предназначен для воспроизведения.
""")]
    bootstrap = '''# @title Подключить папку проекта
USE_DRIVE = True # @param {type:"boolean"}
PROJECT_DIR = "/content/drive/MyDrive/houseplants" # @param {type:"string"}

from pathlib import Path
import shutil

if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')

SOURCE_PROJECT = Path(PROJECT_DIR)
required = [
    'src/pipeline.py', 'configs/experiment.json', 'requirements-colab.txt',
    'prompts/system.txt', 'prompts/judge.txt',
    'data/train.jsonl', 'data/validation.jsonl', 'data/test.jsonl',
]
missing = [name for name in required if not (SOURCE_PROJECT / name).is_file()]
if missing:
    raise FileNotFoundError(
        f'В папке {SOURCE_PROJECT} не найдены: {missing}. '
        'Загрузите dist/houseplants в Мой диск или исправьте PROJECT_DIR.'
    )

# Для скорости код и входные данные копируются в локальную среду Colab.
# Результаты обучения сохраняются отдельно в Drive.
PROJECT = Path('/content/houseplants')
PROJECT.mkdir(parents=True, exist_ok=True)
if SOURCE_PROJECT.resolve() != PROJECT.resolve():
    for folder in ['src', 'configs', 'prompts', 'data', 'tests', 'scripts', 'report', 'results']:
        source = SOURCE_PROJECT / folder
        if source.is_dir():
            shutil.copytree(source, PROJECT / folder, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['README.md', 'PLAN.md', 'MODEL_CARD.md',
                 'requirements-colab.txt', 'requirements-dev.txt']:
        source = SOURCE_PROJECT / name
        if source.is_file():
            shutil.copy2(source, PROJECT / name)
print('Проект подготовлен:', PROJECT)
'''
    cells += [cell("code", bootstrap)]
    cells += [cell("code", """# @title Сохранение результатов
from pathlib import Path
import sys
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from src.pipeline import fingerprint
if USE_DRIVE:
    RUN_ROOT = Path('/content/drive/MyDrive/houseplants_runs')
else:
    RUN_ROOT = Path('/content/houseplants_runs')
RUN_DIR = RUN_ROOT / fingerprint()[:12]
RUN_DIR.mkdir(parents=True, exist_ok=True)
fixed_revisions = PROJECT / 'results/experiment/model_revisions.json'
if fixed_revisions.exists():
    import json, shutil
    revision_file = RUN_DIR / 'model_revisions.json'
    if revision_file.exists():
        if json.loads(revision_file.read_text()) != json.loads(fixed_revisions.read_text()):
            raise ValueError('Ревизии в папке результатов отличаются от сохранённого эксперимента.')
    else:
        shutil.copy2(fixed_revisions, revision_file)
print('Результаты и checkpoints:', RUN_DIR)
""")]
    cells += [cell("code", """# @title Установка фиксированных зависимостей
import subprocess, sys, os
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# Неиспользуемые расширения старого runtime могут конфликтовать с закреплённым torch.
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchvision', 'torchaudio', 'torchtext', 'librosa'], check=False)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '-r', str(PROJECT / 'requirements-colab.txt')], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=False)
""")]
    download_code = (ROOT / "scripts/model_downloads.py").read_text(encoding="utf-8")
    cells += [cell("code", """# @title Подготовить файлы моделей: докачка и повтор при обрыве
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '30'
""" + download_code.replace("from __future__ import annotations\n", "") + "\nprefetch_models(RUN_DIR, PROJECT)\n")]
    cells += [cell("code", """# @title Полный эксперимент: выполнение и возобновление
import subprocess, sys
try:
    subprocess.run([sys.executable, '-m', 'src.pipeline', '--run-dir', str(RUN_DIR)], cwd=PROJECT, check=True)
finally:
    print('Результаты, включая диагностический архив при ошибке:', RUN_DIR)
    archive = RUN_DIR / 'houseplants_results.zip'
    if archive.exists():
        from IPython.display import FileLink, display
        display(FileLink(str(archive)))
""")]
    cells += [cell("code", """# @title Скачать результаты
from google.colab import files
archive = RUN_DIR / 'houseplants_results.zip'
if archive.exists():
    files.download(str(archive))
else:
    print('Архива ещё нет. Проверьте ошибку выше; папка:', RUN_DIR)
""")]
    cells += [cell("markdown", """## После выполнения

Сохраните `houseplants_results.zip` в исходную папку проекта на компьютере и сообщите об этом в чате. По этим реальным результатам можно закончить содержательный разбор и выводы. Архив уже содержит адаптер, данные, метрики, графики, отчёт и логи; большие optimizer-checkpoints остаются в Drive для возобновления.

Если стадия упала: повторите запуск. При систематической ошибке скачайте диагностический архив из папки результатов в Drive и передайте его в папку проекта. Для нового эксперимента с изменённым кодом или настройками используется другой хеш папки, чтобы не смешивать результаты.

Перед сдачей в LMS нужны доступные преподавателю ссылки на репозиторий, данные и веса. Публикация и загрузка в LMS этим ноутбуком не выполняются.
""")]
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}, "colab": {"name": "houseplants_sft.ipynb"}, "accelerator": "GPU"}, "cells": cells}
    for index, item in enumerate(cells):
        item["id"] = f"houseplants-{index}"
    path = ROOT / "notebooks/colab_pipeline.ipynb"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    bundle = dist / "houseplants"
    for source in project_files():
        target = bundle / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(path, bundle / "colab_pipeline.ipynb")
    # ZIP остаётся дополнительным способом передачи папки; ноутбук его не читает.
    with zipfile.ZipFile(dist / "houseplants_project.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(bundle.rglob("*")):
            if source.is_file():
                archive.write(source, source.relative_to(dist))
    print(f"Notebook: {path} ({path.stat().st_size:,} bytes)")
    print(f"Upload folder: {bundle}")


if __name__ == "__main__":
    build()
