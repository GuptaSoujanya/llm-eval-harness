from typing import List, Optional
from .base import BaseLLMProvider, GenerationOutput

class HuggingFaceProvider(BaseLLMProvider):
    """Hugging Face Transformers inference pipeline wrapper."""

    def __init__(self, model_name: str, device: str = "auto", torch_dtype: str = "auto", **kwargs):
        super().__init__(model_name, **kwargs)
        self.device = device
        self.torch_dtype = torch_dtype
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            try:
                import torch
                from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
                self._pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device_map=self.device,
                    torch_dtype=getattr(torch, self.torch_dtype) if hasattr(torch, self.torch_dtype) else "auto"
                )
            except ImportError:
                raise ImportError("Please install transformers and torch: pip install '.[transformers]'")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0, **kwargs) -> GenerationOutput:
        self._load_pipeline()
        do_sample = temperature > 0.0
        res = self._pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            return_full_text=False
        )
        generated_text = res[0]["generated_text"]
        return GenerationOutput(
            text=generated_text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(generated_text.split())
        )
