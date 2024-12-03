from dataclasses import dataclass
from typing import List, Optional, Union

from torch import Tensor


@dataclass
class StepResponse:
    next_state: Optional[Tensor] = None
    label: Optional = None
    terminal: Optional[bool] = None
    info: Optional[dict] = None


@dataclass
class DataLoaderResponse:
    state: Optional[Tensor] = None
    state_labels: Optional[Union[Tensor, List]] = None


@dataclass
class LearningImageLoaderResponse(DataLoaderResponse):
    contaminant_indices: Optional[Tensor] = None
    state_one_hot_labels: Optional[Tensor] = None
    state_soft_labels: Optional[Tensor] = None
    mixture_ratios: Optional[Tensor] = None
