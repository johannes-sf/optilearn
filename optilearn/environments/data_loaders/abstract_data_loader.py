from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
from torch import Tensor

from optilearn.environments.data_loaders.transformations import AbstractTransformationPipeline
from optilearn.environments.responses import DataLoaderResponse
from optilearn.utils.dataset.metadata.dto import TrainingMetaData


class AbstractDataLoader(ABC):
    def __init__(
        self,
        data_path: str = None,
        learning_mode: bool = True,
        shuffle: bool = True,
        batch_size: int = None,
        torch_interface: bool = True,
        device: str = "cpu",
        **kwargs,
    ):
        self.data_path = data_path
        self.learning_mode = learning_mode
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.torch_interface = torch_interface
        self.device = device

        self._current_sample: Union[Tensor, None] = None
        self._current_label = None
        self._current_iteration: int = 0
        self._total_iterations: int = 0
        self._iterator = None
        self.terminal = False

        self.meta_data: TrainingMetaData = None
        self.transformation_pipeline: AbstractTransformationPipeline = None

    def conform_to_interface(self, data: Union[np.array, torch.Tensor]):
        conformed_data = data
        already_torch = isinstance(data, torch.Tensor)

        if self.torch_interface:
            if not already_torch:
                conformed_data = torch.from_numpy(data.astype("float32")).to(self.device)

        else:
            if already_torch:
                conformed_data = data.detach().numpy()

        return conformed_data

    @abstractmethod
    def step(self) -> DataLoaderResponse:
        pass

    def to_tensor(self, x) -> Tensor:
        pass

    @abstractmethod
    def reset(self):
        pass

    @property
    def current_iteration(self) -> int:
        return self._current_iteration

    @property
    def total_iterations(self) -> int:
        return self._total_iterations

    @property
    def current_label(self):
        return self._current_label

    @property
    def current_sample(self) -> Tensor:
        return self._current_sample

    @property
    def iterator(self):
        return self._iterator
