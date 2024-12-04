import re
from typing import List, Optional
from .base import BaseTask, Sample, EvalResult
from ..metrics.accuracy import exact_match_score, normalize_numeric_string

GSM8K_FEWSHOT_PROMPT = """Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
A: Natalia sold 48 / 2 = 24 clips in May.
Altogether she sold 48 + 24 = 72 clips.
#### 72

Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
A: Weng earns 12 / 60 = $0.2 per minute.
Working 50 minutes, she earned 0.2 * 50 = $10.
#### 10

"""

SAMPLE_GSM8K_DATASET = [
    {
        "id": "gsm8k_001",
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins with four. The remainder she sells at the market for $2 each. How much money does she make every day?",
        "answer": "16 - 3 - 4 = 9 eggs remaining.\n9 * 2 = 18 dollars.\n#### 18"
    },
    {
        "id": "gsm8k_002",
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take for 3 robes?",
        "answer": "White fiber per robe = 2 / 2 = 1 bolt.\nTotal per robe = 2 + 1 = 3 bolts.\nFor 3 robes = 3 * 3 = 9 bolts.\n#### 9"
    },
    {
        "id": "gsm8k_003",
        "question": "James decides to run 3 miles every day. He runs each mile in 8 minutes. If he runs every day for 2 weeks, how many total minutes did he spend running?",
        "answer": "Daily minutes = 3 * 8 = 24 minutes.\n2 weeks = 14 days.\nTotal minutes = 24 * 14 = 336 minutes.\n#### 336"
    },
    {
        "id": "gsm8k_004",
        "question": "Each pack of cards contains 15 cards. Mark bought 4 packs. He gave 10 cards to his brother and lost 5 cards. How many cards does he have left?",
        "answer": "Total cards = 4 * 15 = 60.\nRemaining = 60 - 10 - 5 = 45.\n#### 45"
    }
]

class GSM8KTask(BaseTask):
    """Grade School Math 8K (GSM8K) benchmark implementation."""

    @property
    def name(self) -> str:
        return "gsm8k"

    def load_samples(self) -> List[Sample]:
        samples = []
        for idx, item in enumerate(SAMPLE_GSM8K_DATASET):
            prompt = ""
            if self.num_fewshot > 0:
                prompt += GSM8K_FEWSHOT_PROMPT
            prompt += f"Q: {item['question']}\nA:"
            
            target = item['answer'].split("####")[-1].strip()
            samples.append(Sample(id=item["id"], prompt=prompt, target=target))
            if self.max_samples and len(samples) >= self.max_samples:
                break
        return samples

    def extract_answer(self, raw_output: str) -> str:
        # Check standard GSM8K delimiter ####
        if "####" in raw_output:
            return raw_output.split("####")[-1].strip()
        # Fallback to regex for "The answer is X" or last number
        match = re.findall(r"[-+]?\d*\.?\d+", raw_output)
        if match:
            return match[-1]
        return raw_output.strip()

    def evaluate(self, samples: List[Sample], outputs: List[str]) -> EvalResult:
        total = len(samples)
        correct = 0
        details = []

        for sample, raw_out in zip(samples, outputs):
            pred = self.extract_answer(raw_out)
            score = exact_match_score(pred, sample.target, numeric=True)
            correct += score
            details.append({
                "id": sample.id,
                "target": sample.target,
                "extracted_prediction": pred,
                "is_correct": bool(score),
                "raw_output": raw_out
            })

        accuracy = correct / total if total > 0 else 0.0
        return EvalResult(
            task_name=self.name,
            total_samples=total,
            metrics={"accuracy": accuracy, "exact_match": accuracy},
            sample_outputs=details
        )
