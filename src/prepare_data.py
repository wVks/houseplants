"""Детерминированная сборка авторских синтетических примеров, без внешнего API."""
from __future__ import annotations

import difflib
import re
from collections import Counter

from .checks import validate_format
from .common import ROOT, digest, read_json, read_jsonl, sha256, write_json, write_jsonl

TOPICS = ["watering", "light", "wet", "dry", "repot", "feed", "pests", "winter", "propagate", "specific", "uncertain", "drainage"]


def advice(plant, topic):
    light, water, soil = (plant[key] for key in ["light", "water", "soil"])
    if topic == "watering":
        return ("Полив выбирают по состоянию субстрата и растения, а не по одному календарю.", [water, "Перед поливом оцените влажность в глубине и вес горшка; после полива слейте лишнюю воду."], "Каковы температура, освещение, размер горшка и субстрат?", [water, "Проверка глубины важнее фиксированного графика."], ["Безусловно поливать по фиксированному расписанию независимо от влажности."])
    if topic == "light":
        return ("Сначала оцените свет и меняйте размещение постепенно: удобрение не заменяет освещение.", [light, "Понаблюдайте за новым ростом и появлением светлых сухих пятен; при дефиците естественного света рассмотрите досветку."], "Какое окно, расстояние до него и сколько прямого солнца получает растение?", [light, "Резкая смена освещения может повредить растение."], ["Держать растение без света постоянно.", "Резко выставить неадаптированное растение на палящее солнце."])
    if topic == "wet":
        return ("При постоянно мокром субстрате увядание не доказывает жажду: возможны нарушения работы корней.", ["Не добавляйте воду, пока не проверите состояние кома; слейте застойную воду и проверьте отверстия.", "Осмотрите корни при признаках гниения: мягкие разрушающиеся ткани отличаются от упругих здоровых. При повреждении нужна санитарная пересадка с удалением поражённых тканей чистым инструментом.", soil], "Есть ли запах, размягчение основания, какие корни и как долго грунт остаётся сырым?", ["На фоне сырости нельзя автоматически увеличивать полив.", "Корневую гниль нельзя подтвердить только по пожелтению; нужен осмотр."], ["Без проверки добавить ещё воды в постоянно сырой субстрат.", "Считать любое увядание доказательством пересушки."])
    if topic == "dry":
        return ("Сухой лёгкий ком и потеря упругости допускают пересушку, но состояние основания и корней тоже важно.", ["Если ком действительно просох и нет признаков гниения, равномерно увлажните подходящий субстрат, затем дайте всей лишней воде стечь.", water, "Если упругость не возвращается или ткани мягкие и темнеют, проверьте корни; не повторяйте полив бесконечно."], "Насколько глубоко сухо, есть ли мягкие тёмные ткани и свободный сток воды?", [water, "Увядание оценивают вместе с влажностью, а не отдельно."], ["Постоянно оставлять горшок в воде после полива."])
    if topic == "repot":
        return ("Пересадку определяют по состоянию корней и субстрата; чрезмерно большой горшок повышает риск длительной сырости.", ["Выберите устойчивую ёмкость лишь немного больше корневого кома, со свободными сливными отверстиями.", soil, "Осмотрите корни, бережно обращайтесь со здоровыми; не заглубляйте основание сильнее прежнего без видовой необходимости."], "Какой размер корневого кома и есть ли повреждения или признаки разложения субстрата?", [soil, "Сильно увеличивать размер горшка без учёта корневого кома не требуется."], ["Посадить фаленопсис в плотную обычную землю." ] if plant["id"] == "orchid" else ["Обязательно взять огромный горшок без дренажных отверстий."])
    if topic == "feed":
        return ("Удобрение не лечит повреждённые корни, пересушку или недостаток света; избыток может усугубить проблему.", ["Сначала исправьте полив, свет и состояние корней; при стрессе или отсутствии активного роста не увеличивайте подкормку.", "Для здорового активно растущего растения выбирайте подходящее удобрение и соблюдайте инструкцию производителя, не превышая дозу.", "Не наносите концентрат на сухие корни; учитывайте наличие удобрения в свежем субстрате."], "Есть ли активный рост, когда была пересадка, чем и как часто подкармливали?", ["Избыточное удобрение может повредить корни.", "Подкормка не заменяет свет и исправление ошибок полива."], ["Увеличить концентрацию удобрения для лечения повреждённых корней."])
    if topic == "pests":
        return ("По описанию можно заподозрить вредителей, но вид нужно уточнить осмотром.", ["Изолируйте подозрительное растение и осмотрите соседние; проверьте нижнюю сторону листьев и пазухи с увеличением.", "По возможности механически удалите видимых вредителей; для подходящих листьев используйте аккуратное промывание, не заливая субстрат и розетку.", "Сфотографируйте признаки и определите вредителя до выбора средства. При необходимости используйте разрешённый для комнатных растений препарат строго по этикетке."], "Есть ли паутинка, липкость, движущиеся точки, ватные комочки или твёрдые щитки?", ["Изоляция и осмотр предшествуют подбору обработки.", "Средство и дозу нельзя назначить достоверно без определения вредителя."], ["Смешать бытовые химикаты и нанести концентрат на все растения."])
    if topic == "winter":
        return ("Зимний уход зависит от света, температуры и скорости просыхания, а не только от названия сезона.", [light, water, "Защитите от холодного стекла, сквозняков и горячего потока батареи. При замедлении роста уменьшите подкормки; при досветке оценивайте фактический рост."], "Какова температура у горшка, есть ли досветка и за сколько дней просыхает ком?", [water, "В прохладе и при слабом свете субстрат часто просыхает медленнее.", "Нельзя сохранять частоту полива без проверки фактической влажности."], ["Безусловно увеличить полив сырого холодного кома."])
    if topic == "propagate":
        return ("Способ размножения зависит от вида и наличия нужных органов у посадочного материала.", [plant["propagate"], "Используйте здоровый материал и чистый инструмент. Укоренение требует времени; избегайте постоянного заболачивания."], "Есть ли узел, собственные корни или точка роста у выбранной части растения?", [plant["propagate"]], ["Любой одиночный лист любого вида обязательно даст полноценный куст."])
    if topic == "specific":
        return ("Один внешний признак не всегда означает болезнь или нехватку питания.", [plant["quirk"], "Оцените динамику: старое повреждение обычно не восстанавливается, важнее состояние новых тканей и появление новых симптомов.", "Проверьте свет, влажность субстрата, состояние корней и наличие вредителей до резкого изменения ухода."], "Когда возник признак, затронуты ли новые листья и что менялось в условиях?", [plant["quirk"], "По одному симптому нельзя уверенно установить единственную причину."], ["Без осмотра объявить точный инфекционный диагноз и лечить концентратом."])
    if topic == "uncertain":
        return ("По столь краткому описанию нельзя надёжно установить причину: нужны условия и характер симптомов.", ["Уточните, какие части изменились и как быстро; по возможности сделайте фото растения, основания и субстрата.", "Проверьте влажность кома и свободный сток, осмотрите листья на вредителей. До уточнения не меняйте сразу все параметры и не вносите повышенные дозы препаратов."], "Какой свет, температура, частота полива, субстрат, размер горшка и когда появилась проблема?", ["Нужны данные об условиях, влажности и характере симптомов.", "Уверенный диагноз без достаточных сведений необоснован."], ["Уверенно назвать единственную причину без дополнительных данных."])
    if topic == "drainage":
        return ("Слой камней не заменяет отверстия: лишняя вода должна выходить из корневой зоны.", ["Используйте внутренний горшок со сливными отверстиями, поставленный в декоративное кашпо.", "После полива дайте воде стечь и опорожните наружное кашпо или поддон; не держите корни постоянно в воде.", soil], "Есть ли внутренний горшок со сливом и остаётся ли он погружённым в воду?", ["Керамзит не заменяет свободный отвод воды.", "Застой воды вокруг корней нежелателен."], ["Камни полностью заменяют слив, можно постоянно держать корни в воде."])
    raise ValueError(topic)


def render(parts):
    summary, actions, clarify, _, _ = parts
    return f"Краткий ответ:\n{summary}\n\nРекомендации:\n" + "\n".join("- " + action for action in actions) + f"\n\nЧто уточнить:\n{clarify}"


def build():
    profiles = read_json(ROOT / "data/profiles.json")
    system = (ROOT / "prompts/system.txt").read_text().strip()
    splits = {"train": [], "validation": [], "test": []}
    sources = []
    for index, plant in enumerate(profiles):
        source_id = "ncsu_" + plant["id"]
        sources.append({"id": source_id, "title": plant["latin"] + " — NC State Extension", "url": "https://plants.ces.ncsu.edu/plants/" + plant["slug"] + "/", "accessed": "2026-11-05", "use": "Краткие факты; исходный текст статьи в датасет не скопирован."})
        for number, topic in enumerate(TOPICS):
            offset = (number - index) % len(TOPICS)
            split = "validation" if offset == 0 else "test" if offset in (3, 6, 9) else "train"
            group = f"{plant['id']}_{topic}"
            question = plant["questions"][number]
            parts = advice(plant, topic)
            answer = render(parts)
            assert validate_format(answer)["passed"], group
            for variant in range(3 if split == "train" else 1):
                prompt = question
                if variant == 1:
                    prompt = "Помогите разобраться с уходом. " + question
                elif variant == 2:
                    prompt = question + " Подскажите порядок действий и нужные уточнения."
                row = {"id": f"{group}_v{variant}", "group_id": group, "plant": plant["name"], "topic": topic, "question": prompt, "source_ids": [source_id] + (["uc_pests"] if topic == "pests" else ["ncsu_containers"]), "origin": "assistant_authored_rule_composition", "reference_facts": parts[3], "critical_mistakes": parts[4], "reference_answer": answer}
                if topic == "propagate":
                    auxiliary = {"orchid": "rhs_phalaenopsis", "aloe": "illinois_offsets", "monstera": "umn_monstera", "ficus": "mu_propagation", "spider": "illinois_offsets"}.get(plant["id"], "mu_propagation")
                    row["source_ids"].append(auxiliary)
                row["messages"] = [{"role": "system", "content": system}, {"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
                splits[split].append(row)
    sources += [
        {"id": "ncsu_containers", "title": "NC State Extension: Plants Grown in Containers", "url": "https://content.ces.ncsu.edu/extension-gardener-handbook/18-plants-grown-in-containers", "accessed": "2026-11-05"},
        {"id": "uc_pests", "title": "UC IPM: Houseplant Problems", "url": "https://ipm.ucanr.edu/home-and-landscape/houseplant-problems/", "accessed": "2026-11-05"},
        {"id": "mu_propagation", "title": "University of Missouri: Home Propagation of Houseplants", "url": "https://extension.missouri.edu/publications/g6560", "accessed": "2026-11-05"},
        {"id": "rhs_phalaenopsis", "title": "RHS: Growing Phalaenopsis", "url": "https://www.rhs.org.uk/plants/phalaenopsis/growing-guide", "accessed": "2026-11-05"},
        {"id": "illinois_offsets", "title": "Illinois Extension: Specialized Structures", "url": "https://extension.illinois.edu/houseplants/specialized-structures", "accessed": "2026-11-05"},
        {"id": "umn_monstera", "title": "University of Minnesota: Monstera deliciosa", "url": "https://extension.umn.edu/garden-and-home/yard-and-garden/gardening-in-minnesota/propagating-monstera-deliciosa", "accessed": "2026-11-05"},
    ]
    for split, rows in splits.items():
        write_jsonl(ROOT / f"data/{split}.jsonl", rows)
    write_jsonl(ROOT / "data/sources.jsonl", sources)
    return audit()


def normalise(question):
    return re.sub(r"\W+", " ", question.lower()).strip()


def audit():
    splits = {split: read_jsonl(ROOT / f"data/{split}.jsonl") for split in ("train", "validation", "test")}
    source_ids = {row["id"] for row in read_jsonl(ROOT / "data/sources.jsonl")}
    seen_ids, groups, prompts = set(), {}, {}
    stats = {}
    for split, rows in splits.items():
        if not rows:
            raise ValueError("Пустой split: " + split)
        for row in rows:
            if row["id"] in seen_ids:
                raise ValueError("Дубликат id: " + row["id"])
            seen_ids.add(row["id"])
            group = row["group_id"]
            if group in groups and groups[group] != split:
                raise ValueError("Утечка группы: " + group)
            groups[group] = split
            question = normalise(row["question"])
            if question in prompts:
                raise ValueError("Повтор вопроса: " + row["id"])
            prompts[question] = split
            if not set(row["source_ids"]) <= source_ids:
                raise ValueError("Неизвестный источник")
            if not validate_format(row["reference_answer"])["passed"]:
                raise ValueError("Неверный формат эталона")
            if [m["role"] for m in row["messages"]] != ["system", "user", "assistant"]:
                raise ValueError("Некорректные роли")
        stats[split] = {"examples": len(rows), "groups": len({row["group_id"] for row in rows}), "topics": dict(Counter(row["topic"] for row in rows)), "plants": dict(Counter(row["plant"] for row in rows)), "sha256": sha256(ROOT / f"data/{split}.jsonl")}
    train = [row for row in splits["train"] if row["id"].endswith("v0")]
    near = []
    for split in ("validation", "test"):
        for row in splits[split]:
            best = max(train, key=lambda other: difflib.SequenceMatcher(None, normalise(row["question"]), normalise(other["question"])).ratio())
            ratio = difflib.SequenceMatcher(None, normalise(row["question"]), normalise(best["question"])).ratio()
            if ratio > 0.82:
                near.append({"split": split, "id": row["id"], "train_id": best["id"], "similarity": round(ratio, 4)})
    result = {"splits": stats, "cross_split_near_questions": near, "dataset_id": digest(stats), "limitations": ["Синтетический набор с общими шаблонами ответов во всех частях.", "Три формулировки train не являются тремя независимыми ситуациями.", "Источники проверены при подготовке; независимая ботаническая экспертиза не проводилась.", "Тест измеряет перенос между комбинациями растение/задача внутри заданного набора, а не качество на всех реальных вопросах."]}
    write_json(ROOT / "results/data_audit.json", result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    result = audit() if args.audit_only else build()
    print({key: (value["examples"], value["groups"]) for key, value in result["splits"].items()})
    print("Похожих вопросов между частями:", len(result["cross_split_near_questions"]))
