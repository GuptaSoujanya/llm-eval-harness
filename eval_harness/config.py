from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class EvalConfig:
    model_name: str
    provider: str = "mock"
    tasks: List[str] = field(default_factory=lambda: ["gsm8k"])
    batch_size: int = 8
    max_samples: Optional[int] = None
    temperature: float = 0.0
    max_tokens: int = 512
    device: str = "auto"
