import unittest
from eval_harness.metrics.pass_at_k import estimate_pass_at_k

class TestPassAtK(unittest.TestCase):
    def test_all_correct(self):
        self.assertEqual(estimate_pass_at_k(10, 10, 1), 1.0)
        self.assertEqual(estimate_pass_at_k(10, 10, 5), 1.0)

    def test_none_correct(self):
        self.assertEqual(estimate_pass_at_k(10, 0, 1), 0.0)

    def test_partial(self):
        score = estimate_pass_at_k(10, 5, 1)
        self.assertAlmostEqual(score, 0.5, places=2)

if __name__ == "__main__":
    unittest.main()
