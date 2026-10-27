from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class GenerationOutput:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

class BaseLLMProvider(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> GenerationOutput:
        pass
    def generate_batch(self, prompts: List[str], **kwargs) -> List[GenerationOutput]:
        return [self.generate(p, **kwargs) for p in prompts]

    # Greedy decoding token guard
