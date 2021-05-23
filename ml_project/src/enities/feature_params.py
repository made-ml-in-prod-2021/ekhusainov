from dataclasses import dataclass, field
from typing import List, Optional


@dataclass()
class FeatureParams:
    categorial_features: List[str]
    numerical_features: List[str]
    target_column: Optional[str]
