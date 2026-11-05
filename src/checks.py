"""Проверка структуры. Не оценивает ботаническую правильность текста."""
from __future__ import annotations

import re

HEADINGS = ("Краткий ответ", "Рекомендации", "Что уточнить")


def heading(line):
    line = re.sub(r"^\s*#{1,6}\s*", "", line).strip()
    line = line.replace("**", "").replace("__", "").strip()
    return line.rstrip(":").strip()


def validate_format(answer):
    errors = []
    if not isinstance(answer, str) or not answer.strip():
        return {"passed": False, "errors": ["empty_answer"], "recommendation_count": 0}
    lines = answer.splitlines()
    positions = [(index, heading(line)) for index, line in enumerate(lines) if heading(line) in HEADINGS]
    observed = [value for _, value in positions]
    for value in HEADINGS:
        if observed.count(value) != 1:
            errors.append("heading_count:" + value)
    if observed != list(HEADINGS):
        errors.append("heading_order")
    sections = {}
    for i, (start, value) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(lines)
        sections[value] = lines[start + 1:end]
        if not any(line.strip() for line in sections[value]):
            errors.append("empty_section:" + value)
    recommendations = sections.get("Рекомендации", [])
    bullets = [line for line in recommendations if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+\S", line)]
    if not 2 <= len(bullets) <= 5:
        errors.append("recommendations_count")
    return {"passed": not errors, "errors": errors, "recommendation_count": len(bullets)}


def examples():
    valid = "Краткий ответ:\nПричин может быть несколько.\n\nРекомендации:\n- Проверьте влажность субстрата.\n- Осмотрите дренажные отверстия.\n\nЧто уточнить:\nКакое растение и когда был последний полив?"
    return [
        {"name": "correct", "answer": valid, "expected": True},
        {"name": "markdown", "answer": valid.replace("Краткий ответ:", "### **Краткий ответ:**"), "expected": True},
        {"name": "missing_section", "answer": valid.split("Что уточнить:")[0], "expected": False},
        {"name": "empty_section", "answer": valid.replace("Причин может быть несколько.", ""), "expected": False},
        {"name": "prose_instead_of_list", "answer": valid.replace("- ", ""), "expected": False},
        {"name": "duplicate_heading", "answer": valid + "\nЧто уточнить:\nЕщё вопрос.", "expected": False},
        {"name": "wrong_order", "answer": valid.replace("Краткий ответ:", "TEMP").replace("Что уточнить:", "Краткий ответ:").replace("TEMP", "Что уточнить:"), "expected": False},
    ]


if __name__ == "__main__":
    from .common import ROOT, write_json
    rows = [{**case, "actual": validate_format(case["answer"])} for case in examples()]
    assert all(row["actual"]["passed"] == row["expected"] for row in rows)
    write_json(ROOT / "results/check_examples.json", rows)
    print("Проверено примеров:", len(rows))
