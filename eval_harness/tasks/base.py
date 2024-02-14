from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

@dataclass
class Sample:
    id: str
    prompt: str
    target: str

@dataclass
class EvalResult:
    task_name: str
    total_samples: int
    metrics: dict
