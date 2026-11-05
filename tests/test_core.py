import tempfile
import unittest
from pathlib import Path

from src.checks import examples, validate_format
from src.common import read_jsonl, write_jsonl
from src.evaluate import paired_interval, score_index, summarise
from src.judge import parse_score
from src.modeling import CompletionCollator, encode_example


class FormatTests(unittest.TestCase):
    def test_documented_examples(self):
        for case in examples():
            with self.subTest(case=case["name"]):
                self.assertEqual(case["expected"], validate_format(case["answer"])["passed"])

    def test_empty_bullets_are_not_actions(self):
        value = examples()[0]["answer"].replace("- Проверьте влажность субстрата.", "- ")
        self.assertFalse(validate_format(value)["passed"])

    def test_empty_answer(self):
        self.assertFalse(validate_format("")["passed"])


class JudgeTests(unittest.TestCase):
    valid = '{"correctness": 4, "usefulness": 5, "context": 3, "consistency": 5, "critical_error": false, "reason": "Уместно."}'

    def test_extract_markdown_json(self):
        self.assertEqual(parse_score("```json\n" + self.valid + "\n```")["correctness"], 4)

    def test_invalid_scores_are_not_accepted(self):
        for value in [self.valid.replace('"correctness": 4', '"correctness": true'), self.valid.replace('"context": 3', '"context": 6'), self.valid.replace('false', '"false"'), "{}", "not json"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_score(value)


class MetricTests(unittest.TestCase):
    def test_missing_generations_and_judge_scores(self):
        valid = examples()[0]["answer"]
        expected = [{"id": "a", "plant": "A", "topic": "wet"}, {"id": "b", "plant": "B", "topic": "wet"}]
        outputs = {"baseline": {"a": {"answer": valid}}, "sft": {"a": {"answer": valid}, "b": {"answer": valid}}}
        score = parse_score(JudgeTests.valid)
        judged = {("baseline", "a"): score, ("sft", "a"): score, ("sft", "b"): score}
        result = summarise(expected, outputs, judged, 20)
        self.assertEqual(result["metrics"]["baseline"]["format_pass_rate"], 0.5)
        self.assertEqual(result["metrics"]["baseline"]["generation_errors"], 1)
        self.assertEqual(result["metrics"]["sft"]["judge_common_n"], 1)
        self.assertEqual(result["paired_deltas"]["correctness"]["mean"], 0)
        self.assertEqual(result["metrics"]["baseline"]["judge_coverage"], 0.5)

    def test_repeats_do_not_change_main_score(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            write_jsonl(path, [{"variant": "baseline", "id": "a", "repeat": False, "score": {"correctness": 2}}, {"variant": "baseline", "id": "a", "repeat": True, "score": {"correctness": 5}}])
            self.assertEqual(score_index(path)[("baseline", "a")]["correctness"], 2)

    def test_paired_bootstrap_preserves_constant_difference(self):
        self.assertEqual(paired_interval([1, 1, 1], ["a", "a", "b"], 50), [1, 1])
        self.assertIsNone(paired_interval([1], ["a"], 50))


class ToyTokenizer:
    eos_token_id = 3
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1, 2, 4] if add_generation_prompt else [1, 2, 4, 7, 8, 3, 9]


class TokenTests(unittest.TestCase):
    def test_loss_excludes_prompt_and_includes_end(self):
        row = {"id": "toy", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
        encoded = encode_example(row, ToyTokenizer(), 10)
        self.assertEqual(encoded["labels"], [-100, -100, -100, 7, 8, 3, 9])
        with self.assertRaises(ValueError):
            encode_example(row, ToyTokenizer(), 4)


if __name__ == "__main__":
    unittest.main()
