from __future__ import annotations

import argparse
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path

from .checks import validate_format
from .common import ROOT, config, read_jsonl, write_json
from .judge import CRITERIA


def paired_interval(differences, clusters, samples=2000, seed=42):
    """Парный cluster bootstrap по растениям; разности сохраняются внутри пары."""
    grouped = defaultdict(list)
    for value, cluster in zip(differences, clusters):
        grouped[cluster].append(value)
    if len(grouped) < 2:
        return None
    names = sorted(grouped)
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        selected = [value for name in rng.choices(names, k=len(names)) for value in grouped[name]]
        values.append(statistics.mean(selected))
    values.sort()
    return [values[int((samples-1)*0.025)], values[int((samples-1)*0.975)]]


def score_index(path):
    return {(row["variant"], row["id"]): row["score"] for row in read_jsonl(path) if not row.get("repeat") and row.get("score")}


def summarise(expected, outputs, judges, bootstrap_samples=2000):
    per_case, metrics = [], {}
    for case in expected:
        entry = {"id": case["id"], "plant": case["plant"], "topic": case["topic"]}
        for variant in ["baseline", "sft"]:
            row = outputs.get(variant, {}).get(case["id"])
            failed = not row or bool(row.get("error")) or not row.get("answer", "").strip()
            check = validate_format(row.get("answer", "") if row and not failed else "")
            entry[variant] = {"format_pass": int(check["passed"]), "format_errors": check["errors"], "generation_error": failed, "truncated": bool(row.get("truncated")) if row else False, "score": judges.get((variant, case["id"])) if not failed else None}
        per_case.append(entry)
    if not per_case:
        raise ValueError("Нельзя оценить пустой test")
    common = [case for case in per_case if all(case[v]["score"] for v in ["baseline", "sft"])]
    for variant in ["baseline", "sft"]:
        scores = [case[variant]["score"] for case in per_case if case[variant]["score"]]
        metrics[variant] = {"n_expected": len(per_case), "format_pass_rate": statistics.mean(case[variant]["format_pass"] for case in per_case), "generation_errors": sum(case[variant]["generation_error"] for case in per_case), "truncated": sum(case[variant]["truncated"] for case in per_case), "judge_n_valid": len(scores), "judge_coverage": len(scores)/len(per_case), "judge_common_n": len(common)}
        for key in CRITERIA:
            metrics[variant][key] = statistics.mean(case[variant]["score"][key] for case in common) if common else None
        metrics[variant]["critical_error_rate"] = statistics.mean(case[variant]["score"]["critical_error"] for case in common) if common else None
    deltas = {}
    for name in ["format_pass_rate", *CRITERIA, "critical_error_rate"]:
        cases = per_case if name == "format_pass_rate" else common
        def value(case, variant):
            if name == "format_pass_rate":
                return case[variant]["format_pass"]
            return case[variant]["score"]["critical_error" if name == "critical_error_rate" else name]
        differences = [value(case, "sft")-value(case, "baseline") for case in cases]
        deltas[name] = {"mean": statistics.mean(differences) if differences else None, "ci95": paired_interval(differences, [case["plant"] for case in cases], samples=bootstrap_samples), "n": len(cases)}
    return {"metrics": metrics, "paired_deltas": deltas, "per_case": per_case}


def chart(path, labels, series, maximum, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(9, max(2.5, len(labels)*0.8)))
    for j, (name, values) in enumerate(series):
        positions = [i + (j-0.5)*0.32 for i in range(len(labels))]
        available = [(position, value) for position, value in zip(positions, values) if value is not None]
        axis.barh([x for x, _ in available], [value for _, value in available], height=0.3, label=name, color=["#64748b", "#22855a"][j])
    axis.set(yticks=list(range(len(labels))), yticklabels=labels, xlim=(0, maximum), title=title)
    axis.legend()
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def loss_chart(path, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(8, 3.5))
    for key, label in [("loss", "train loss"), ("eval_loss", "validation loss")]:
        rows = [row for row in history if key in row]
        axis.plot([row["step"] for row in rows], [row[key] for row in rows], marker="o", label=label)
    axis.set(xlabel="Шаг оптимизатора", ylabel="Cross-entropy", title="Ход обучения")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def evaluate(run_dir):
    expected = read_jsonl(ROOT / "data/test.jsonl")
    outputs = {variant: {row["id"]: row for row in read_jsonl(run_dir / f"{variant}_test.jsonl")} for variant in ["baseline", "sft"]}
    judges = score_index(run_dir / "judge_test.jsonl")
    summary = summarise(expected, outputs, judges, config()["bootstrap_samples"])
    summary["status"] = "measured" if all(summary["metrics"][v]["generation_errors"] == 0 and summary["metrics"][v]["judge_coverage"] == 1 for v in ["baseline", "sft"]) else "incomplete"
    repeats = [row for row in read_jsonl(run_dir / "judge_test.jsonl") if row.get("repeat") and row.get("score")]
    repeat_deltas = []
    for row in repeats:
        original = judges.get((row["variant"], row["id"]))
        if original:
            repeat_deltas.append(statistics.mean(abs(row["score"][key]-original[key]) for key in CRITERIA))
    summary["judge_repeat_mean_absolute_difference"] = statistics.mean(repeat_deltas) if repeat_deltas else None
    summary["judge_repeat_n"] = len(repeat_deltas)
    out = run_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary)
    with (out / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", *summary["metrics"]["baseline"]])
        writer.writeheader()
        for variant, values in summary["metrics"].items():
            writer.writerow({"variant": variant, **values})
    chart(out / "format.svg", ["Доля ответов в формате"], [(v, [summary["metrics"][v]["format_pass_rate"]]) for v in ["baseline", "sft"]], 1, "Автоматическая проверка")
    chart(out / "judge.svg", list(CRITERIA), [(v, [summary["metrics"][v][key] for key in CRITERIA]) for v in ["baseline", "sft"]], 5, "Модель-судья: общий набор валидных пар")
    if (run_dir / "training.json").exists():
        from .common import read_json
        loss_chart(out / "loss.svg", read_json(run_dir / "training.json")["log_history"])
    write_json(out / "by_topic.json", {topic: summarise([case for case in expected if case["topic"] == topic], outputs, judges, 200)["metrics"] for topic in sorted({case["topic"] for case in expected})})
    print("Оценка:", summary["status"], out, flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    evaluate(parser.parse_args().run_dir)
