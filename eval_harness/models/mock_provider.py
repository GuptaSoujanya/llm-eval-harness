import re
from typing import List, Optional
from .base import BaseLLMProvider, GenerationOutput

class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic mock provider for unit testing, offline validation, and CI workflows.
    Generates realistic synthetic responses for GSM8K, MMLU, and HumanEval.
    """

    def __init__(self, model_name: str = "mock-qwen-7b", accuracy_rate: float = 0.85, **kwargs):
        super().__init__(model_name, **kwargs)
        self.accuracy_rate = accuracy_rate
        self.call_count = 0

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0, **kwargs) -> GenerationOutput:
        self.call_count += 1
        prompt_lower = prompt.lower()

        # Simulated GSM8K output
        if "gsm8k" in prompt_lower or "math problem" in prompt_lower or "####" in prompt:
            # Check if there is an expected answer encoded or generate synthetic reasoning
            answer_val = (self.call_count * 17) % 100
            text = f"Step 1: Calculate the intermediate balance.\nStep 2: Add remaining parts.\nThe total is {answer_val}.\n#### {answer_val}"
            return GenerationOutput(text=text, prompt_tokens=len(prompt.split()), completion_tokens=35)

        # Simulated MMLU output
        if "question:" in prompt_lower and "a)" in prompt_lower:
            choices = ["A", "B", "C", "D"]
            choice = choices[self.call_count % 4]
            text = f"The correct answer is ({choice})."
            return GenerationOutput(text=text, prompt_tokens=len(prompt.split()), completion_tokens=10)

        # Simulated HumanEval output
        if "def " in prompt:
            text = "    return sum(x for x in numbers if x > 0)\n"
            return GenerationOutput(text=text, prompt_tokens=len(prompt.split()), completion_tokens=15)

        # Default fallback
        return GenerationOutput(
            text=f"Mock response from {self.model_name} for input of length {len(prompt)}.",
            prompt_tokens=len(prompt.split()),
            completion_tokens=20
        )
