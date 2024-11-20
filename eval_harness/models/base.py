from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class GenerationOutput:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: Optional[str] = "stop"
    raw_response: Optional[Dict[str, Any]] = None

class BaseLLMProvider(ABC):
    """Abstract base interface for all LLM inference backends."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0, **kwargs) -> GenerationOutput:
        """Generate completion for a single prompt."""
        pass

    def generate_batch(self, prompts: List[str], max_tokens: int = 512, temperature: float = 0.0, **kwargs) -> List[GenerationOutput]:
        """Generate completions for a batch of prompts (fallback sequential loop)."""
        return [self.generate(p, max_tokens=max_tokens, temperature=temperature, **kwargs) for p in prompts]
