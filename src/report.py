from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from .common import ROOT, config, now, read_json, read_jsonl
from .judge import CRITERIA


def format_value(value):
    return "не измерено" if value is None else f"{value:.3f}"


def build_report(run_dir=None):
    audit = read_json(ROOT / "results/data_audit.json")
    cfg = config()
    measured = run_dir is not None and (run_dir / "evaluation/summary.json").exists()
    summary = read_json(run_dir / "evaluation/summary.json") if measured else None
    training = read_json(run_dir / "training.json") if run_dir and (run_dir / "training.json").exists() else None
    sections = ["# Дообучение языковой модели: домашние растения", f"Сформировано: {now()}.\n\nСтатус: **" + ("есть реальные измерения, полнота: " + summary["status"] if measured else "подготовка завершена; основной GPU-эксперимент ещё не выполнен") + "**. Незаполненные результаты не заменяются предположениями."]
    sections += ["## 1. Постановка задачи и гипотеза", "Задача — адаптировать русскоязычного помощника по уходу за домашними растениями. Вход: текстовый вопрос с видом растения и условиями. Выход: краткий ответ, 2–5 рекомендаций и необходимые уточнения. При неполных данных помощник должен выражать неопределённость, а не выдумывать точный диагноз.\n\nГипотеза: тематический SFT улучшит соблюдение структуры и практическую полезность. Проверяем фактическую корректность отдельно: красивый ответ не равен правильному. Исследование ограничено текстом, без определения растения по фото, RAG и пользовательского интерфейса."]
    sections += ["## 2. Модель, протокол и первые ответы", f"База: `{cfg['model_id']}`. Метод: LoRA поверх 4-битного NF4 с double quantization. Сравниваются одна ревизия базы без адаптера и с адаптером при одинаковом квантовании, системном промпте и greedy decoding. Лимит ответа — {cfg['max_new_tokens']} токенов. Ревизии фиксируются в `model_revisions.json` при первом GPU-запуске.\n\nПервые два вопроса берутся из validation, а не из test. Исходные сообщения и необработанные ответы сохраняются в `baseline_validation.jsonl`; пробный судья оценивает их до основного обучения."]
    pilot = read_jsonl(run_dir / "baseline_validation.jsonl")[:2] if run_dir else []
    if pilot:
        for row in pilot:
            sections += [f"**Вопрос:** {row['question']}\n\n**Исходный ответ:**\n\n" + row["answer"]]
            pilot_scores = [entry["score"] for entry in read_jsonl(run_dir / "judge_validation.jsonl") if entry["id"] == row["id"] and entry["variant"] == "baseline" and entry.get("score")]
            if pilot_scores:
                score = pilot_scores[0]
                sections += ["**Пробная оценка судьи:** " + ", ".join(f"{key}={score[key]}" for key in CRITERIA) + ". " + score["reason"]]
    else:
        sections += ["Первый прогон ещё не выполнен. Код запуска подготовлен; сгенерированные моделью ответы на этом этапе отсутствуют."]
    sections += ["## 3. Автоматическая проверка и модель-судья", "Проверка `validate_format` требует три непустых раздела в установленном порядке и 2–5 пунктов рекомендаций. Допускается Markdown у заголовков. Она выявляет ошибки структуры, но не распознаёт неверные ботанические советы. Это полезно для воспроизводимого измерения понятного пользователю порядка действий.\n\nПроверка протестирована на корректном ответе, Markdown, пропуске и повторе раздела, пустом разделе, нарушении порядка и отсутствии списка. Реальные результаты этих локальных проверок находятся в `results/check_examples.json`."]
    examples = read_json(ROOT / "results/check_examples.json")
    sections += ["| Пример | Ожидание | Результат |\n| --- | --- | --- |\n" + "\n".join(f"| {row['name']} | {row['expected']} | {row['actual']['passed']} |" for row in examples)]
    sections += ["Отдельный судья — `Qwen/Qwen2.5-7B-Instruct` в 4-битном режиме. Для каждого ответа он видит вопрос, опорные факты и существенные ошибки; отвечающая модель факты из оценочной записи не получает. Баллы 1–5: correctness, usefulness, context, consistency. Отдельно — critical_error и объяснение. Версии ответов скрыты от судьи, задания перемешаны с seed=42. Проверяется JSON, при ошибке формата допускается одна попытка исправления; невалидные оценки отмечаются как пропуски.\n\nШесть заданий повторяются с обратным порядком полей в контексте: содержание ответа и фактов сохраняется. Это проверяет чувствительность к размещению контекста; простой повтор идентичного greedy-запроса был бы малоинформативен. Повторы — не независимые судьи. Одно семейство у базы и судьи создаёт возможную систематическую предвзятость. Автоматический судья не заменяет ботаническую экспертизу; его оценки требуют содержательного разбора."]
    sections += ["## 4. Данные и контроль утечек", "Подготовлен авторский синтетический набор из 12 профилей растений и 12 типов задач. Краткие факты сверены с NC State Extension и UC IPM, ссылки хранятся в `data/sources.jsonl`. Полные статьи не копировались. Вопросы составлены отдельно для каждой комбинации; ответы собраны программно из фактов и правил. Происхождение явно отмечено как `assistant_authored_rule_composition`. Независимая экспертная разметка не проводилась."]
    sections += ["| Часть | Примеров | Исходных ситуаций |\n| --- | ---: | ---: |\n" + "\n".join(f"| {name} | {values['examples']} | {values['groups']} |" for name, values in audit["splits"].items())]
    sections += ["В train для каждой ситуации три варианта формулировки. Это аугментация, не три независимых примера. Вместо первоначального ориентира на 1000 записей выбран меньший прозрачный набор: увеличение числа почти одинаковых строк не расширяет покрытие фактов.\n\nРазбиение детерминировано до обучения: для каждого растения 8 ситуаций train, 1 validation и 3 test. Ни одна группа вариантов не пересекает части. Проверены уникальность id и вопросов, источники, роли и формат эталонов. Дополнительно сопоставлены вопросы validation/test с train по символьной близости; результат аудита сохранён.\n\nОбщие шаблоны ответов и тематики остаются во всех частях. Поэтому тест измеряет перенос между комбинациями растения и задачи внутри синтетического распределения, а не универсальную консультативную способность. Метрики на 36 тестовых ситуациях имеют большую неопределённость."]
    for row in read_jsonl(ROOT / "data/train.jsonl")[:1] + read_jsonl(ROOT / "data/test.jsonl")[:1]:
        sections += [f"**Пример данных `{row['id']}`:** {row['question']}\n\n{row['reference_answer']}\n\nИсточник: {', '.join(row['source_ids'])}."]
    sections += ["## 5. Обучение и принятые решения", f"Используется Transformers Trainer и PEFT. Вместо TRL реализована явная маска: токены system/user имеют label=-100, loss считается только на assistant, включая токен завершения. Это упрощает проверку границы ответа и не зависит от специальных generation-маркеров chat template. Чрезмерно длинные примеры не обрезаются молча: код останавливается с их id.\n\nПараметры: rank={cfg['lora_r']}, alpha={cfg['lora_alpha']}, dropout={cfg['lora_dropout']}, все линейные слои; learning rate={cfg['learning_rate']}, batch={cfg['batch_size']}, накопление={cfg['gradient_accumulation_steps']}, эпох={cfg['epochs']}, длина={cfg['max_length']}, FP16, gradient checkpointing, cosine schedule. Checkpoint выбирается по validation loss; test при выборе не используется. Это проще первоначального плана отбора по генеративным оценкам и уменьшает число длительных прогонов."]
    sections += ["Перед основным запуском выполняются два шага на T4 с сохранением пробного адаптера. Полный процесс запускает модели последовательно в разных процессах: GPU-память освобождается перед загрузкой судьи. Checkpoints, логи и ответы пишутся в постоянную папку; завершённые этапы и отдельные генерации можно продолжить. Хеши исходного кода, данных, промптов и настроек защищают от смешения результатов разных конфигураций."]
    if training:
        sections += [f"**Реальный запуск:** GPU — {training['gpu']}; время — {training['seconds']/60:.1f} мин; пиковая выделенная память — {training['peak_allocated_gb']:.2f} GiB; лучший validation loss — {format_value(training['best_eval_loss'])}. Полная история в `training.json`."]
        if run_dir and (run_dir / "evaluation/loss.svg").exists():
            sections += ["![Loss](evaluation/loss.svg)"]
    else:
        sections += ["Основное обучение на T4 пока не запускалось. Веса целевого адаптера, время T4 и реальные training/validation loss ещё не получены. Проверка маленькой случайной CPU-модели проверяет код, а не качество Qwen2.5-3B; её результаты не включаются в сравнение качества."]
    sections += ["## 6. Количественная оценка до и после", "Автоматическая доля соблюдения формата использует все тестовые вопросы в знаменателе, включая ошибки генерации. Для судьи публикуется покрытие; средние качества и разности считаются на общем наборе валидно оценённых пар. Пропущенная оценка не заменяется нулём и не исчезает из показателя покрытия. Помимо общего результата сохраняются тематические срезы.\n\nРазности имеют 95%-интервалы парного cluster bootstrap по растениям. Из-за малого синтетического набора, общих шаблонов и корреляции между темами эти интервалы описательны и не доказывают перенос на реальных пользователей."]
    if measured:
        lines = ["| Метрика | До SFT | После SFT | Разность | 95% интервал |", "| --- | ---: | ---: | ---: | --- |"]
        for key in ["format_pass_rate", *CRITERIA, "critical_error_rate"]:
            delta = summary["paired_deltas"][key]
            ci = delta["ci95"]
            lines.append(f"| {key} | {format_value(summary['metrics']['baseline'][key])} | {format_value(summary['metrics']['sft'][key])} | {format_value(delta['mean'])} | {('[' + ', '.join(format_value(x) for x in ci) + ']') if ci else 'недостаточно данных'} |")
        sections += ["\n".join(lines)]
        for variant, metrics in summary["metrics"].items():
            sections += [f"{variant}: ошибки генерации={metrics['generation_errors']}; обрезанные ответы={metrics['truncated']}; валидные оценки судьи={metrics['judge_n_valid']}/{metrics['n_expected']}; общих пар={metrics['judge_common_n']}."]
        sections += [f"Повторных оценок судьи: {summary['judge_repeat_n']}; среднее абсолютное расхождение: {format_value(summary['judge_repeat_mean_absolute_difference'])}.", "![Формат](evaluation/format.svg)\n\n![Оценки судьи](evaluation/judge.svg)"]
    else:
        sections += ["Метрики до/после пока **не измерены**. Таблица и графики будут построены автоматически после реального GPU-запуска. Числа из тестов кода не используются как результаты эксперимента."]
    sections += ["## 7. Качественный анализ и ошибки"]
    if measured:
        outputs = {variant: {row["id"]: row for row in read_jsonl(run_dir / f"{variant}_test.jsonl")} for variant in ["baseline", "sft"]}
        def change(case):
            if all(case[v]["score"] for v in ["baseline", "sft"]):
                return statistics.mean(case["sft"]["score"][key]-case["baseline"]["score"][key] for key in CRITERIA)
            return 0
        ordered = sorted(summary["per_case"], key=change)
        candidates = ordered[:3] + list(reversed(ordered[-3:]))
        used = set()
        for case in candidates:
            if case["id"] in used:
                continue
            used.add(case["id"])
            first = outputs["baseline"].get(case["id"], {})
            second = outputs["sft"].get(case["id"], {})
            sections += [f"### Ситуация {case['id']}", "**Вопрос:** " + first.get("question", second.get("question", "нет генерации")), "**До:**\n\n" + first.get("answer", "ошибка генерации"), "**После:**\n\n" + second.get("answer", "ошибка генерации")]
            for variant in ["baseline", "sft"]:
                score = case[variant]["score"]
                sections += [f"Оценка {variant}: " + (score["reason"] if score else "нет валидной оценки") + f"; ошибки структуры: {case[variant]['format_errors']}."]
        sections += ["Пары отобраны автоматически по изменению оценки: крайние улучшения и ухудшения. Это объяснения модели-судьи, а не независимая экспертная проверка. После получения архива требуется содержательно проверить исходные ответы и опорные факты, особенно все critical_error; список для этого находится в `evaluation/summary.json`."]
    else:
        sections += ["Исходные ошибки целевой модели ещё не наблюдались. Примеры неправильного формата в разделе 3 специально составлены для проверки программы и не выдаются за ответы Qwen. После запуска этот раздел будет содержать реальные пары ответов и объяснения судьи."]
    sections += ["## 8. Выводы, ограничения и воспроизведение"]
    if measured:
        delta = summary["paired_deltas"]["format_pass_rate"]
        sections += [f"Наблюдаемое изменение соблюдения формата: {format_value(delta['mean'])}. Изменение корректности на общем наборе оценённых пар: {format_value(summary['paired_deltas']['correctness']['mean'])}. Положительная разность одной метрики не означает общего улучшения. Сначала проверяются покрытие, интервалы и существенные ошибки, затем формулируется предметный вывод."]
    else:
        sections += ["На текущем этапе подтверждена подготовка к эксперименту, а не улучшение модели. Сформированы данные, протокол, проверка формата и код. Проверка целевого обучения и сравнение остаются зависимыми от запуска на GPU."]
    sections += ["Ограничения: небольшой синтетический датасет, шаблонное происхождение ответов, отсутствие независимой ботанической экспертизы, один судья из того же семейства, небольшая validation-часть, отсутствие многократного обучения с разными seed. Улучшение может ограничиться форматом. Дальнейшая работа — реальные разнообразные вопросы и независимая оценка, а не простое увеличение количества шаблонных перефразировок.\n\nВоспроизведение: открыть `notebooks/colab_pipeline.ipynb`, выбрать T4 и выполнить все ячейки. Ноутбук содержит исходный проект, устанавливает фиксированные версии, запускает этапы и сохраняет архив. Потребность в памяти и работоспособность именно T4 подтверждаются только реальным Colab-запуском. Полные версии среды сохраняются в `environment.json` и `pip-freeze.txt`.\n\nАртефакты после запуска: `adapter/adapter_model.safetensors`, конфигурация и токенизатор, исходные JSONL, генерации, промпты и ответы судьи, CSV-метрики, SVG-графики, логи, настройки и ревизии моделей. Ссылки на репозиторий и веса добавляются после публикации пользователем; в LMS загружаются ссылка и отчёт."]
    sections += ["## Источники", "\n".join(f"- [{row['title']}]({row['url']})" for row in read_jsonl(ROOT / "data/sources.jsonl")), "- [Базовая модель Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)\n- [Судья Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)\n- [PEFT: квантование и LoRA](https://huggingface.co/docs/peft/developer_guides/quantization)\n- [Transformers Trainer](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/trainer)"]
    destination = run_dir / "report.md" if run_dir else ROOT / "report/report.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print("Отчёт:", destination, flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    build_report(parser.parse_args().run_dir)
