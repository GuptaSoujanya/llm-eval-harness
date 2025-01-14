from typing import List, Optional, Dict, Any
from .base import BaseTask, Sample, EvalResult
from ..metrics.pass_at_k import estimate_pass_at_k

SAMPLE_HUMANEVAL_DATASET = [
    {
        "task_id": "HumanEval/0",
        "prompt": "def has_close_elements(numbers, threshold):\n    \"\"\" Check if elements are close. \"\"\"\n",
        "canonical_solution": "    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n",
        "entry_point": "has_close_elements",
        "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n"
    },
    {
        "task_id": "HumanEval/1",
        "prompt": "def separate_paren_groups(paren_string):\n    \"\"\" Separate paren groups. \"\"\"\n",
        "canonical_solution": "    res = []\n    cur = []\n    d = 0\n    for c in paren_string:\n        if c == '(': d += 1; cur.append(c)\n        elif c == ')':\n            d -= 1; cur.append(c)\n            if d == 0: res.append(''.join(cur)); cur = []\n    return res\n",
        "entry_point": "separate_paren_groups",
        "test": "def check(candidate):\n    assert candidate('( ) (( ))') == ['()', '(())']\n"
    }
]

class HumanEvalTask(BaseTask):
    """OpenAI HumanEval coding benchmark."""

    @property
    def name(self) -> str:
        return "humaneval"

    def load_samples(self) -> List[Sample]:
        samples = []
        for item in SAMPLE_HUMANEVAL_DATASET:
            samples.append(Sample(
                id=item["task_id"],
                prompt=item["prompt"],
                target=item["canonical_solution"],
                metadata={"entry_point": item["entry_point"], "test": item["test"]}
            ))
            if self.max_samples and len(samples) >= self.max_samples:
                break
        return samples

    def extract_answer(self, raw_output: str) -> str:
        return raw_output

    def evaluate(self, samples: List[Sample], outputs: List[str]) -> EvalResult:
        total = len(samples)
        passed = 0
        details = []

        for sample, raw_out in zip(samples, outputs):
            full_code = sample.prompt + "\n" + raw_out + "\n" + sample.metadata["test"] + f"\ncheck({sample.metadata['entry_point']})\n"
            is_success = False
            error_msg = None

            try:
                exec_globals = {}
                exec(full_code, exec_globals)
                is_success = True
                passed += 1
            except Exception as e:
                error_msg = str(e)

            details.append({
                "id": sample.id,
                "passed": is_success,
                "error": error_msg,
                "completion": raw_out
            })

        pass_1 = estimate_pass_at_k(total_samples=total, num_correct=passed, k=1)
        accuracy = passed / total if total > 0 else 0.0

        return EvalResult(
            task_name=self.name,
            total_samples=total,
            metrics={"pass@1": pass_1, "success_rate": accuracy},
            sample_outputs=details
        )
