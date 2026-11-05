# Домашние растения: воспроизводимое SFT на T4

**Built with Qwen.** Учебный исследовательский проект; условия использования модели — в [лицензии базы](artifacts/adapter/BASE_MODEL_LICENSE.txt).

Проект по заданию из `ЗАДАНИЕ`: baseline, автоматическая проверка, отдельная модель-судья, данные SFT, LoRA, сравнение до/после и отчёт.

[Открыть ноутбук в Google Colab](https://colab.research.google.com/github/wVks/houseplants/blob/main/notebooks/colab_pipeline.ipynb)

## Ссылки для сдачи

- Репозиторий: [github.com/wVks/houseplants](https://github.com/wVks/houseplants).
- [Итоговый отчёт PDF — 9 страниц](https://github.com/wVks/houseplants/blob/main/report/report.pdf).
- [Папка обученного LoRA-адаптера](https://github.com/wVks/houseplants/tree/main/artifacts/adapter), [скачать safetensors](https://media.githubusercontent.com/media/wVks/houseplants/main/artifacts/adapter/adapter_model.safetensors).
- Данные JSONL: [train](https://raw.githubusercontent.com/wVks/houseplants/main/data/train.jsonl), [validation](https://raw.githubusercontent.com/wVks/houseplants/main/data/validation.jsonl), [test](https://raw.githubusercontent.com/wVks/houseplants/main/data/test.jsonl).

Веса хранятся через Git LFS. Для загрузки всего проекта вместе с адаптером:

```bash
git lfs install
git clone https://github.com/wVks/houseplants.git
cd houseplants
git lfs pull
```

При загрузке ZIP с GitHub файл весов может оказаться LFS-указателем. В таком случае скачайте safetensors по прямой ссылке выше в `artifacts/adapter/adapter_model.safetensors`. Проверенный SHA256: `cf59e34d24fba2b10dc971fcfdf4f809def7effb042b087615c4882353ed2924`.

## Соответствие ЗАДАНИЮ

| Требование | Артефакт |
| --- | --- |
| Тема и постановка задачи | [Отчёт, раздел 1](report/report.md), [исходное задание](ЗАДАНИЕ.docx) |
| 1–2 первых вопроса до обучения | [Первые ответы baseline](results/experiment/baseline_validation.jsonl), отчёт §1 |
| Автоматическая проверка, обоснование и срабатывания | [Код проверки](src/checks.py), [7 примеров](results/check_examples.json), отчёт §3 |
| Проба другой модели-судьи | [Код](src/judge.py), [рубрика](prompts/judge.txt), [пилотные и validation-оценки](results/experiment/judge_validation.jsonl) |
| Данные для SFT в совместимом формате | [Train JSONL](data/train.jsonl), [описание](data/README.md), [источники](data/sources.jsonl) |
| Реальное дообучение на T4 | [Код](src/train.py), [конфигурация](configs/experiment.json), [история обучения](results/experiment/training.json), [лог](results/experiment/logs/08_training.log) |
| Оценка до и после | [CSV метрик](results/experiment/evaluation/metrics.csv), [интервалы и оценки](results/experiment/evaluation/summary.json), [разбор 36 пар](report/appendix_answers.md) |
| Веса safetensors и конфигурация | [Обученный адаптер](artifacts/adapter/), [проверка целостности](results/final_validation.json) |
| Таблицы, графики и логи | [Результаты эксперимента](results/experiment/), [графики](report/figures/) |
| Отчёт 5–10 страниц с выводами | [PDF, 9 страниц](report/report.pdf), [Markdown](report/report.md) |
| Воспроизведение в Colab | [Ноутбук](notebooks/colab_pipeline.ipynb), инструкции ниже; фактический запуск на T4 подтверждён логами |

**Текущий статус:** эксперимент завершён на Tesla T4. Обучен и проверен LoRA-адаптер, получены ответы до/после, оценки судьи, метрики и итоговый отчёт. Все 15 стадий завершились. Историческая ошибка скачивания в сыром `state.json` относится к предыдущей попытке; итоговый статус — `computed`.

## Готовые результаты

- [Итоговый отчёт PDF, 9 страниц](report/report.pdf) и [Markdown](report/report.md).
- [Все 36 пар ответов с разбором](report/appendix_answers.md).
- [Метрики](results/experiment/evaluation/metrics.csv), [полные оценки и интервалы](results/experiment/evaluation/summary.json), [логи и конфигурации](results/experiment/).
- Адаптер: `artifacts/adapter/adapter_model.safetensors`, 504 тензора, 29 933 568 параметров; SHA256 — в `results/final_validation.json`.
- [Что сдать в LMS](SUBMISSION.md). Данные находятся в `data/`, код — в `src/`, короткий ноутбук — в `notebooks/`.

| Показатель на 36 вопросах | До SFT | После SFT |
| --- | ---: | ---: |
| Соблюдение формата | 31/36 = 86,11% | 36/36 = 100% |
| Корректность по модели-судье | 3,39/5 | 3,83/5 |
| Полезность по модели-судье | 4,47/5 | 4,67/5 |
| Флаг critical_error судьи | 7/36 | 1/36 |

Подтверждён рост соблюдения структуры на данном тесте. Общий рост предметного качества не доказан: интервалы качественных разностей содержат ноль; в ряде объяснений судьи обнаружены приписанные утверждения. Число его флагов не равно установленной частоте опасных советов. Содержательный разбор всех 36 пар — в `results/qualitative_review.json`; независимой ботанической экспертизы нет.

### Проверить сохранённый эксперимент без обучения

Без GPU, установки пакетов и загрузки моделей:

```bash
python3 scripts/verify_submission.py --skip-weights
```

С проверкой safetensors, если установлены зависимости разработки:

```bash
.venv/bin/python scripts/verify_submission.py
```

В полном комплекте для сдачи веса лежат рядом с папкой `repository`: из неё используйте `--adapter-dir ../adapter`. Подтверждены точный пересчёт метрик и bootstrap-интервалов, исходные JSON судьи, ревизии моделей, отсутствие assistant-ответа в промптах генерации и конечность значений весов. Это проверка сохранённых артефактов; она не запускает заново всю 3B-модель на CPU.

## Самый короткий запуск

В исходной рабочей папке комплект для Drive уже подготовлен. Если вы скачали только репозиторий, сначала выполните `python3 scripts/build_notebook.py`: эта команда использует стандартную библиотеку и создаст `dist/houseplants`.

1. В [Google Drive](https://drive.google.com/) откройте **Мой диск → Создать → Загрузить папку** и выберите подготовленную папку **`dist/houseplants`**. В ней только нужные файлы: код, данные и небольшой ноутбук. [Инструкция Google](https://support.google.com/drive/answer/2424368?hl=ru).
2. В загруженной папке откройте **`colab_pipeline.ipynb`** через Colab. Выберите **Runtime → Change runtime type → T4 GPU**, затем **Runtime → Run all**. При запросе подтвердите подключение Google Drive.
3. Дождитесь завершения и скачайте `houseplants_results.zip`. Автоматический отчёт в новом запуске содержит исходные оценки; итоговый разбор текущего эксперимента уже находится в `report/report.pdf`.

Код и JSONL находятся в обычных файлах внутри загруженной папки; в ноутбуке нет встроенного архива и длинной строки Base64. Путь по умолчанию — `/content/drive/MyDrive/houseplants`; при другом расположении измените `PROJECT_DIR` в первой ячейке. Для скорости входные файлы копируются в локальную среду Colab, результаты сохраняются в Drive.

Загружайте именно `dist/houseplants`: в ней нет локальной `.venv` и кэшей. GitHub и API-ключи не нужны. Внешние модели скачиваются с Hugging Face: более 20 GB для базы и судьи. Время зависит от T4, сети и доступности Colab; оценку скорости можно сделать по первым этапам.

Результаты сохраняются в `MyDrive/houseplants_runs/<хеш>/`. При разрыве сессии повторите Run all: pipeline продолжит с незавершённой стадии, а генерации — с необработанного вопроса. Если CPU-ноутбук сообщает об отсутствии CUDA, переключите тип runtime. Если Drive не подключён, отключение сессии уничтожит локальные checkpoints; по умолчанию Drive включён.

Перед pipeline ноутбук готовит кэш моделей: скачивает по одному файлу, при временном обрыве сети повторяет до четырёх попыток и использует незавершённые части из кэша Hugging Face. Сетевой таймаут повышен до 120 секунд. Ошибки доступа и нехватка места не маскируются повторами. Эта подготовка не меняет ревизии моделей и идентификатор эксперимента; уже выполненные этапы сохраняются.

Если старый ноутбук остановился с `IncompleteRead` / `ChunkedEncodingError`, сначала повторно запустите ячейку «Полный эксперимент» в том же сеансе — Hugging Face использует кэш. Для ограниченных автоматических повторов можно добавить в текущий ноутбук код из `scripts/colab_download_recovery.py`, выполнить его и повторить «Полный эксперимент». Новая версия ноутбука уже содержит такую подготовку.

Установка заменяет версии библиотек в одноразовой Colab-среде и удаляет неиспользуемые расширения torchvision/torchaudio/torchtext/librosa, чтобы исключить их несовместимость с закреплённым torch. На компьютере проект запускается из отдельной `.venv`.

## Что выполнит ноутбук

1. Аудит данных и проверку формата с примерами срабатывания.
2. Первые два ответа базовой модели на validation и их оценку отдельной моделью.
3. Два пробных шага SFT, сохранение адаптера и загрузку его отдельным процессом.
4. Полный baseline на validation и test.
5. Две эпохи QLoRA; выбор checkpoint по validation loss.
6. Генерации модели с адаптером на тех же вопросах.
7. Оценку отдельной 7B-моделью без API-ключа, проверку JSON и повторные оценки.
8. CSV, парные разности и интервалы, SVG-графики, подбор примеров и Markdown-отчёт.
9. Архив результатов; при ошибке — диагностический архив с доступными логами.

Каждая модель работает отдельным процессом и освобождает GPU перед следующим этапом. Настройки, данные, промпты и код входят в хеш эксперимента. Изменения требуют новой папки результатов; результаты разных запусков не смешиваются. `model_revisions.json` фиксирует точные commit SHA моделей, `pip-freeze.txt` — реально установленные пакеты.

## Данные и честность оценки

В наборе 12 растений × 12 типов задач = 144 исходные ситуации. Train: 96 ситуаций × 3 варианта формулировки = 288 строк; validation: 12; test: 36. Исходные профили и вопросы — `data/profiles.json`; детерминированный сборщик — `src/prepare_data.py`. Все части совместимы с:

```python
from datasets import load_dataset
dataset = load_dataset('json', data_files={
    'train': 'data/train.jsonl',
    'validation': 'data/validation.jsonl',
    'test': 'data/test.jsonl',
})
```

Данные синтетические и собраны из авторских вопросов и правил на основе кратких фактов из профильных источников. Полные статьи не копировались. Три варианта train не являются независимыми ситуациями. В разных частях остаются общие шаблоны ответов; тест характеризует ограниченное синтетическое распределение. Это существенное ограничение, описанное в отчёте.

Test и его опорные ответы не подаются в Trainer. Генератор получает только system/user. Содержимое тестовых генераций не используется для выбора checkpoint. При пересборке комплекта следует сначала пересобрать данные и локальные отчёты, затем выполнить `python3 scripts/build_notebook.py`: команда создаст короткий ноутбук и папку `dist/houseplants`.

Проверка формата измеряет только структуру. Судья оценивает correctness/usefulness/context/consistency по шкале 1–5 и существенные ошибки. Средние судьи сравниваются на общем наборе валидных пар; покрытие и ошибки публикуются отдельно. Модель-судья из того же семейства может иметь систематическое смещение. Автоматический разбор примеров требует последующей содержательной проверки.

## Настройки и ограничения проверки

База — `Qwen/Qwen2.5-3B-Instruct`, судья — `Qwen/Qwen2.5-7B-Instruct`. Конфигурация: `configs/experiment.json`. Target: T4, FP16, NF4, rank=16, alpha=32, gradient checkpointing, batch=1, accumulation=16, 2 эпохи, learning rate=1e-4. Число эпох и настройки не подбираются по test. Версии прямых зависимостей закреплены в `requirements-colab.txt`; полная Linux-среда фиксируется при запуске.

Локально выполнены 15 поведенческих и интеграционных проверок, включая повтор сетевой загрузки, формирование CSV, графиков, отчёта и архива на временной тестовой фикстуре. Отдельно проверены токенизатор, маски и сохранение LoRA на маленькой CPU-модели. Затем в реальном Colab выполнены пробное обучение, загрузка пробного адаптера, две эпохи основной 3B-модели и полная оценка. Основное обучение заняло 835,4 с, пик выделений PyTorch — 4,37 GiB. Validation loss: 1,061 → 0,831. Доказательства: `results/experiment/training.json`, `results/experiment/logs/`, `results/final_validation.json`.

## Командный запуск на Linux с GPU

```bash
python -m pip install -r requirements-colab.txt
python -m src.pipeline --run-dir /path/to/persistent/houseplants-run
```

По отдельности:

```bash
python -m src.prepare_data --audit-only
python -m src.checks
python -m src.train --run-dir runs/example --smoke
python -m src.train --run-dir runs/example
python -m src.generate --run-dir runs/example --variant baseline --split test
python -m src.generate --run-dir runs/example --variant sft --split test
python -m src.judge --run-dir runs/example --split test
python -m src.evaluate --run-dir runs/example
python -m src.report --run-dir runs/example
```

Для честного полного протокола используйте pipeline: он сам задаёт порядок baseline, обучения и оценки. Папка вывода должна быть доступна на запись. Для повторения чужого запуска скопируйте его `model_revisions.json` в новую папку до старта, используйте тот же исходный проект из архива.

Подготовленный ноутбук автоматически копирует сохранённые ревизии из `results/experiment/model_revisions.json`, если файл есть, перед первым запуском в новой папке.

### Загрузить обученный адаптер на GPU

```python
import json
from src.modeling import load_model, generate_one

model_id = "Qwen/Qwen2.5-3B-Instruct"
revision = json.load(open("results/experiment/model_revisions.json"))[model_id]
model, tokenizer = load_model(model_id, revision, adapter="artifacts/adapter")
system = open("prompts/system.txt").read()
messages = [{"role": "system", "content": system},
            {"role": "user", "content": "Как проверить, пора ли поливать монстеру?"}]
print(generate_one(model, tokenizer, messages, 384)["answer"])
```

Используются GPU и зависимости `requirements-colab.txt`. Веса базы скачиваются отдельно. Адаптер сохраняет лицензионные ограничения базы; качество ограничено учебным экспериментом.

## Проверки разработки

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Для `python -m tests.cpu_integration` нужен настоящий токенизатор в `.cache/qwen-tokenizer/` (два файла `tokenizer.json` и `tokenizer_config.json` из базовой модели). Тест не загружает веса модели. Сборка ноутбука и папки для Drive: `python3 scripts/build_notebook.py`.

Итоговый отчёт текущего реального запуска формируется командой `python scripts/build_final_report.py --run-dir results/experiment`; для отчёта достаточно сохранённых результатов, веса читать не требуется. Готовый PDF уже включён в комплект. Библиотека ReportLab нужна только для PDF и включена в `requirements-dev.txt`.

## Артефакты и сдача

После GPU-запуска архив содержит:

- `adapter/adapter_model.safetensors`, конфигурацию PEFT и токенизатор;
- ответы до/после и полные запросы/ответы судьи;
- `evaluation/metrics.csv`, `summary.json`, тематические срезы и SVG;
- `training.json`, логи стадий, ревизии и сведения о среде;
- `report.md` и копию исходного проекта в `project/`.

Optimizer-checkpoints и пробный адаптер в архив не входят, но остаются в Drive. Для LMS используются ссылка на этот репозиторий и `report/report.pdf` на 9 страницах. Данные и обученный адаптер доступны по ссылкам в начале README.

Лицензия базы 3B — `qwen-research`, не Apache 2.0; её условия распространяются на использование адаптера с базой. Сведения о происхождении данных и ссылки на источники: `data/README.md`, `data/sources.jsonl`. Это учебный эксперимент с ограниченной оценкой.
