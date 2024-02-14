from .base import Sample, EvalResult
from ..metrics.accuracy import exact_match_score

SAMPLE_GSM8K = [
    Sample("gsm8k_1", "Janet's ducks lay 16 eggs per day. She eats 3, bakes 4, sells rest at $2. How much does she make?\nA:", "18"),
    Sample("gsm8k_2", "A robe takes 2 bolts blue fiber and 1 bolt white. How many for 3 robes?\nA:", "9")
]

class GSM8KTask:
    name = "gsm8k"
    def load_samples(self): return SAMPLE_GSM8K
    def evaluate(self, samples, outputs):
        correct = sum(exact_match_score(o.split('####')[-1] if '####' in o else o, s.target) for s, o in zip(samples, outputs))
        acc = correct / len(samples) if samples else 0.0
        return EvalResult(self.name, len(samples), {"accuracy": acc, "exact_match": acc})
