"""Код для репозитория и полный комплект с отдельным адаптером."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sources():
    paths = []
    for folder in ["src","configs","prompts","data","scripts","tests","results","report","notebooks"]:
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                paths.append(path)
    paths += [ROOT / name for name in ["README.md","MODEL_CARD.md","PLAN.md","SUBMISSION.md","requirements-colab.txt","requirements-dev.txt",".gitignore",".gitattributes","ЗАДАНИЕ.docx"]]
    return sorted(set(paths))


def checksum(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def zip_folder(folder, destination):
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(folder.rglob("*")):
            if source.is_file() and "__pycache__" not in source.parts and source.suffix != ".pyc":
                archive.write(source, Path(folder.name) / source.relative_to(folder))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None


def build():
    dist = ROOT / "dist"
    repository = dist / "houseplants_repository"
    repository.mkdir(parents=True, exist_ok=True)
    for source in sources():
        target = repository / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    # Проверка проводится обычным Python без установки ML-пакетов.
    subprocess.run(["python3", "scripts/verify_submission.py", "--skip-weights"], cwd=repository, check=True)
    assert not (repository / "artifacts").exists()
    zip_folder(repository, dist / "houseplants_repository.zip")

    package = dist / "houseplants_submission"
    shutil.copytree(repository, package / "repository", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    shutil.copytree(ROOT / "artifacts/adapter", package / "adapter", dirs_exist_ok=True)
    shutil.copy2(ROOT / "SUBMISSION.md", package / "SUBMISSION.md")
    subprocess.run([sys.executable, "scripts/verify_submission.py", "--adapter-dir", "../adapter"], cwd=package / "repository", check=True)
    manifest = {str(path.relative_to(package)): {"bytes":path.stat().st_size,"sha256":checksum(path)} for path in sorted(package.rglob("*")) if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc" and path.name != "MANIFEST.json"}
    (package / "MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2) + "\n",encoding="utf-8")
    zip_folder(package, dist / "houseplants_submission.zip")
    print("Репозиторий:",repository)
    print("Полный архив:",dist / "houseplants_submission.zip")
    print("Файлов в полном комплекте:",len(manifest))


if __name__ == "__main__":
    build()
