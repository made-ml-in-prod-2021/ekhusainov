from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelParams:
    model_type: str
    penalty: Optional[str] = field(default_factory="l2")
    tol: float = field(default_factory=1e-4)
    C: float = field(default_factory=1.0)
    max_iter: int = field(default_factory="l2")
    random_state: int = field(default_factory=1337)
    n_estimators: int = field(default_factory=100)
    criterion: Optional[str] = field(default_factory="gini")
    max_depth: int = field(default_factory=20)
    min_samples_split: int = field(default_factory=2)