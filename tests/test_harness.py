import unittest
from eval_harness.metrics.accuracy import exact_match_score, normalize_numeric_string
from eval_harness.metrics.pass_at_k import estimate_pass_at_k

class TestHarness(unittest.TestCase):
    def test_norm(self):
        self.assertEqual(normalize_numeric_string("$1,200.00"), "1200")
    def test_pass_at_k(self):
        self.assertEqual(estimate_pass_at_k(10, 10, 1), 1.0)
        self.assertAlmostEqual(estimate_pass_at_k(10, 5, 1), 0.5, places=2)

if __name__ == "__main__":
    unittest.main()
