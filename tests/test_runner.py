import unittest
from eval_harness.config import EvalConfig
from eval_harness.runner import EvaluationRunner

class TestRunner(unittest.TestCase):
    def test_mock_pipeline_execution(self):
        cfg = EvalConfig(
            model_name="mock-model",
            provider="mock",
            tasks=["gsm8k", "mmlu"],
            max_samples=2
        )
        runner = EvaluationRunner(cfg)
        results = runner.run()
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertGreater(r.total_samples, 0)

if __name__ == "__main__":
    unittest.main()
