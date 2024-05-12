from .base import BaseLLMProvider, GenerationOutput

class MockLLMProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "mock-model"):
        super().__init__(model_name)
        self.count = 0
    def generate(self, prompt: str, **kwargs) -> GenerationOutput:
        self.count += 1
        if "gsm8k" in prompt.lower() or "####" in prompt:
            return GenerationOutput(text=f"Reasoning step.\n#### {self.count * 15}", prompt_tokens=len(prompt.split()), completion_tokens=25)
        if "question:" in prompt.lower():
            return GenerationOutput(text="The correct answer is (B).", prompt_tokens=len(prompt.split()), completion_tokens=10)
        return GenerationOutput(text="def solution():\n    return True\n", prompt_tokens=len(prompt.split()), completion_tokens=15)
