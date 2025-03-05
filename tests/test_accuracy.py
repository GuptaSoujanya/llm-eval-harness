import unittest
from eval_harness.metrics.accuracy import exact_match_score, normalize_numeric_string

class TestMetrics(unittest.TestCase):
    def test_normalize_numeric_string(self):
        self.assertEqual(normalize_numeric_string("$1,250.00"), "1250")
        self.assertEqual(normalize_numeric_string("42%"), "42")
        self.assertEqual(normalize_numeric_string("The answer is 75 dollars"), "75")
        self.assertEqual(normalize_numeric_string("-3.14"), "-3.14")

    def test_exact_match_score(self):
        self.assertEqual(exact_match_score("100", "100"), 1.0)
        self.assertEqual(exact_match_score("$100", "100"), 1.0)
        self.assertEqual(exact_match_score("99", "100"), 0.0)
        self.assertEqual(exact_match_score("yes", "Yes", numeric=False), 1.0)

if __name__ == "__main__":
    unittest.main()
