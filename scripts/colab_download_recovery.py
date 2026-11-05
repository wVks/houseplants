# Вставьте этот код в новую ячейку текущего Colab после ошибки скачивания.
# Используются PROJECT и RUN_DIR, уже заданные ячейками вашего ноутбука.
import json
import os
import time

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
from huggingface_hub import snapshot_download
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout

model_id = "Qwen/Qwen2.5-7B-Instruct"
revision = json.loads((RUN_DIR / "model_revisions.json").read_text())[model_id]
for attempt in range(4):
    try:
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
            max_workers=1,
        )
        break
    except (ChunkedEncodingError, ConnectionError, Timeout):
        if attempt == 3:
            raise
        print("Обрыв сети. Повтор скачивания из кэша через 10 секунд.")
        time.sleep(10)
print("Модель скачана. Повторите ячейку «Полный эксперимент».")
