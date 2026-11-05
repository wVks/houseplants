"""Интеграция отчёта на синтетической фикстуре во временной папке, не метрики модели."""
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

from src.common import ROOT, read_jsonl, write_json, write_jsonl
from src.evaluate import evaluate
from src.pipeline import package_results
from src.report import build_report


class ArtifactTests(unittest.TestCase):
    def test_metrics_charts_report_and_archive(self):
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            cases = read_jsonl(ROOT / "data/test.jsonl")
            for variant in ["baseline", "sft"]:
                write_jsonl(run / f"{variant}_test.jsonl", [{"id": case["id"], "question": case["question"], "answer": case["reference_answer"], "scope": "TEST FIXTURE ONLY"} for case in cases])
            judges = [{"job_id": variant + ":" + case["id"], "id": case["id"], "variant": variant, "score": {"correctness": 3, "usefulness": 3, "context": 3, "consistency": 3, "critical_error": False, "reason": "TEST FIXTURE ONLY"}} for case in cases for variant in ["baseline", "sft"]]
            write_jsonl(run / "judge_test.jsonl", judges)
            summary = evaluate(run)
            self.assertEqual(summary["status"], "measured")
            self.assertEqual(summary["metrics"]["baseline"]["format_pass_rate"], 1)
            self.assertEqual(summary["paired_deltas"]["correctness"]["mean"], 0)
            report = build_report(run)
            self.assertIn("TEST FIXTURE ONLY", report.read_text())
            self.assertTrue((run / "evaluation/judge.svg").exists())
            (run / "checkpoints").mkdir()
            (run / "checkpoints/optimizer.pt").write_bytes(b"fixture")
            archive = package_results(run)
            with zipfile.ZipFile(archive) as zipped:
                self.assertIn("report.md", zipped.namelist())
                self.assertIn("project/src/train.py", zipped.namelist())
                self.assertNotIn("checkpoints/optimizer.pt", zipped.namelist())
                self.assertEqual(len(json.loads(zipped.read("evaluation/summary.json"))["per_case"]), 36)


if __name__ == "__main__":
    unittest.main()
