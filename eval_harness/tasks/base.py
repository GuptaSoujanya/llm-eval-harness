from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Sample:
    id: str
    prompt: str
    target: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvalResult:
    task_name: str
    total_samples: int
    metrics: Dict[str, float]
    sample_outputs: List[Dict[str, Any]] = field(default_factory=list)

class BaseTask(ABC):
    """Abstract base class for benchmark tasks."""

    def __init__(self, num_fewshot: int = 0, max_samples: Optional[int] = None):
        self.num_fewshot = num_fewshot
        self.max_samples = max_samples

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def load_samples(self) -> List[Sample]:
        pass

    @abstractmethod
    def extract_answer(self, raw_output: str) -> str:
        pass

    @abstractmethod
    def evaluate(self, samples: List[Sample], outputs: List[str]) -> EvalResult:
        pass
