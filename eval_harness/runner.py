from typing import List, Optional
from .config import EvalConfig
from .models.base import BaseLLMProvider
from .models.mock_provider import MockLLMProvider
from .tasks.base import BaseTask, EvalResult
from .tasks.gsm8k import GSM8KTask
from .tasks.mmlu import MMLUTask
from .tasks.humaneval import HumanEvalTask

TASK_REGISTRY = {
    "gsm8k": GSM8KTask,
    "mmlu": MMLUTask,
    "humaneval": HumanEvalTask
}

class EvaluationRunner:
    """Core evaluation coordinator."""

    def __init__(self, config: EvalConfig, provider: Optional[BaseLLMProvider] = None):
        self.config = config
        self.provider = provider or self._init_provider()

    def _init_provider(self) -> BaseLLMProvider:
        if self.config.provider == "mock":
            return MockLLMProvider(model_name=self.config.model_name)
        elif self.config.provider == "hf":
            from .models.huggingface import HuggingFaceProvider
            return HuggingFaceProvider(model_name=self.config.model_name, device=self.config.device)
        else:
            raise ValueError(f"Unknown provider '{self.config.provider}'")

    def run(self) -> List[EvalResult]:
        results = []
        for task_name in self.config.tasks:
            if task_name not in TASK_REGISTRY:
                raise ValueError(f"Unrecognized task: {task_name}. Available: {list(TASK_REGISTRY.keys())}")
            
            task_cls = TASK_REGISTRY[task_name]
            task: BaseTask = task_cls(max_samples=self.config.max_samples)
            samples = task.load_samples()

            prompts = [s.prompt for s in samples]
            outputs = self.provider.generate_batch(
                prompts,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            raw_texts = [o.text for o in outputs]
            eval_res = task.evaluate(samples, raw_texts)
            results.append(eval_res)

        return results
