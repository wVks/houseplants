# Комплект для сдачи

Вычислительный эксперимент и отчёт готовы. Повторное обучение для получения этих артефактов не требуется.

## Основные файлы

- `report/report.pdf` — итоговый отчёт на 9 страниц.
- `report/report.md` — тот же текст в Markdown.
- `report/appendix_answers.md` — все 36 пар ответов, оценки судьи и содержательные комментарии.
- `results/experiment/` — исходные генерации, оценки, метрики, графики, логи, конфигурация и ревизии.
- `data/` — данные JSONL, профили, источники и описание.
- `artifacts/adapter/` — обученный LoRA в исходной рабочей папке.
- `notebooks/colab_pipeline.ipynb` — воспроизведение в Colab на T4.

Подтверждено улучшение структуры на тесте: 86,11% → 100%. Не заявляется доказанное общее улучшение предметной полезности: содержательные ошибки сохраняются, интервалы широкие, у модели-судьи найдены неподтверждённые обвинения. Этот вывод отражён в отчёте.

## Готовая упаковка

- `dist/houseplants_repository/` — папка с кодом, данными, отчётом и результатами для репозитория, без больших весов.
- `dist/houseplants_repository.zip` — архив этой папки.
- `dist/houseplants_submission.zip` — полный комплект: папка `repository`, отдельная папка `adapter` и эта инструкция.

После распаковки полного комплекта из `repository` можно выполнить:

```bash
python3 scripts/verify_submission.py --skip-weights
```

Для проверки весов после установки зависимостей:

```bash
python scripts/verify_submission.py --adapter-dir ../adapter
```

## Ссылки для LMS

- Репозиторий: [https://github.com/wVks/houseplants](https://github.com/wVks/houseplants).
- Отчёт: [report/report.pdf](https://github.com/wVks/houseplants/blob/main/report/report.pdf), 9 страниц.
- Веса и конфигурация: [artifacts/adapter](https://github.com/wVks/houseplants/tree/main/artifacts/adapter).
- Файл весов: [adapter_model.safetensors](https://media.githubusercontent.com/media/wVks/houseplants/main/artifacts/adapter/adapter_model.safetensors).
- Данные: [data](https://github.com/wVks/houseplants/tree/main/data), формат JSONL.

В LMS нужно загрузить PDF и ссылку на репозиторий. Данные, код, оценки, логи и веса доступны из репозитория. Для клонирования весов используйте Git LFS согласно README. Отправка в LMS остаётся отдельным действием.
