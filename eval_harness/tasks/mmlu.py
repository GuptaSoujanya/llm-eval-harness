from .base import Sample, EvalResult
import re

SAMPLE_MMLU = [
    Sample("mmlu_cs_1", "What is worst case quicksort?\nA) O(n)\nB) O(n^2)\nAnswer:", "B"),
    Sample("mmlu_ml_2", "Which regularizer adds absolute weight penalty?\nA) L1\nB) L2\nAnswer:", "A")
]

class MMLUTask:
    name = "mmlu"
    def load_samples(self): return SAMPLE_MMLU
    def evaluate(self, samples, outputs):
        correct = 0
        for s, o in zip(samples, outputs):
            match = re.search(r'\b([A-D])\b', o.upper())
            pred = match.group(1) if match else o[:1]
            if pred == s.target: correct += 1
        acc = correct / len(samples) if samples else 0.0
        return EvalResult(self.name, len(samples), {"accuracy": acc})
