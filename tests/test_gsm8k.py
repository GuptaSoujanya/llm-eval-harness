import unittest
from eval_harness.tasks.gsm8k import GSM8KTask
from eval_harness.models.mock_provider import MockLLMProvider

class TestGSM8K(unittest.TestCase):
    def setUp(self):
        self.task = GSM8KTask()
        self.provider = MockLLMProvider()

    def test_load_samples(self):
        samples = self.task.load_samples()
        self.assertGreater(len(samples), 0)
        self.assertIn("Q:", samples[0].prompt)

    def test_extract_answer(self):
        raw = "Calculated total is 48.\n#### 48"
        self.assertEqual(self.task.extract_answer(raw), "48")

    def test_evaluate_mock(self):
        samples = self.task.load_samples()
        prompts = [s.prompt for s in samples]
        outputs = [self.provider.generate(p).text for p in prompts]
        res = self.task.evaluate(samples, outputs)
        self.assertEqual(res.task_name, "gsm8k")
        self.assertIn("accuracy", res.metrics)

if __name__ == "__main__":
    unittest.main()
