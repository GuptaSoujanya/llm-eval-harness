import re
from typing import List, Optional
from .base import BaseTask, Sample, EvalResult

SAMPLE_MMLU_DATASET = [
    {
        "id": "mmlu_cs_01",
        "subject": "computer_science",
        "question": "What is the worst-case time complexity of quicksort when using standard Hoare partitioning without randomized pivots?",
        "choices": ["O(n log n)", "O(n)", "O(n^2)", "O(log n)"],
        "answer": "C"
    },
    {
        "id": "mmlu_ml_02",
        "subject": "machine_learning",
        "question": "Which regularization technique specifically adds a penalty proportional to the absolute value of the model weights?",
        "choices": ["L2 (Ridge)", "L1 (Lasso)", "Elastic Net", "Dropout"],
        "answer": "B"
    },
    {
        "id": "mmlu_stat_03",
        "subject": "statistics",
        "question": "According to the Central Limit Theorem, the distribution of sample means approaches what distribution as sample size n grows large?",
        "choices": ["Poisson", "Uniform", "Normal / Gaussian", "Exponential"],
        "answer": "C"
    },
    {
        "id": "mmlu_arch_04",
        "subject": "system_architecture",
        "question": "In modern Transformer architectures, which component allows independent focus on different representation subspaces?",
        "choices": ["Rotary Position Embeddings", "Multi-Head Attention", "SwiGLU Feed-Forward Network", "RMSNorm"],
        "answer": "B"
    }
]

class MMLUTask(BaseTask):
    """Massive Multitask Language Understanding (MMLU) benchmark."""

    def __init__(self, subject: Optional[str] = None, num_fewshot: int = 0, max_samples: Optional[int] = None):
        super().__init__(num_fewshot=num_fewshot, max_samples=max_samples)
        self.subject = subject

    @property
    def name(self) -> str:
        return f"mmlu_{self.subject}" if self.subject else "mmlu"

    def load_samples(self) -> List[Sample]:
        samples = []
        labels = ["A", "B", "C", "D"]
        for item in SAMPLE_MMLU_DATASET:
            if self.subject and item["subject"] != self.subject:
                continue
            
            prompt = f"Question: {item['question']}\n"
            for label, choice in zip(labels, item["choices"]):
                prompt += f"{label}) {choice}\n"
            prompt += "Answer:"

            samples.append(Sample(id=item["id"], prompt=prompt, target=item["answer"], metadata={"subject": item["subject"]}))
            if self.max_samples and len(samples) >= self.max_samples:
                break
        return samples

    def extract_answer(self, raw_output: str) -> str:
        clean = raw_output.strip().upper()
        # Direct pattern matching: "A", "(A)", "The answer is A"
        match = re.search(r"\b([A-D])\b", clean)
        if match:
            return match.group(1)
        return clean[:1] if clean else ""

    def evaluate(self, samples: List[Sample], outputs: List[str]) -> EvalResult:
        total = len(samples)
        correct = 0
        details = []

        for sample, raw_out in zip(samples, outputs):
            pred = self.extract_answer(raw_out)
            is_match = (pred == sample.target.upper())
            if is_match:
                correct += 1
            details.append({
                "id": sample.id,
                "target": sample.target,
                "extracted_prediction": pred,
                "is_correct": is_match,
                "raw_output": raw_out
            })

        acc = correct / total if total > 0 else 0.0
        return EvalResult(
            task_name=self.name,
            total_samples=total,
            metrics={"accuracy": acc},
            sample_outputs=details
        )
