import os
import pathlib
from abc import ABC, abstractmethod
from typing import Dict, Type

import numpy as np
import pandas as pd

from optilearn.configs.constants import RunModes
from optilearn.environments.data_loaders import AbstractDataLoader
from optilearn.environments.data_loaders.transformations import AbstractTransformationPipeline
from optilearn.environments.responses import StepResponse


class AbstractEnv(ABC):
    """ """

    def __init__(
        self,
        config: dict,
        data_loader: Type[AbstractDataLoader] = None,
        transformation_pipeline: Type[AbstractTransformationPipeline] = None,
        run_mode: RunModes = RunModes.TRAINING,
        device: str = "cpu",
    ):
        """
        Base class for all environments

        Parameters
        ----------
        config : dict
                 Dictionary with all arguments needed to setup the environment

        run_mode :
        """
        self.config: Dict = config
        self.torch_interface: bool = config["torch_interface"]
        self.run_mode = run_mode
        self.device = device
        self._data_loader = data_loader
        self.transformation_pipeline = transformation_pipeline
        self.loader: AbstractDataLoader = None
        self.state = None
        self.state_labels = None
        self.terminal = False
        self.steps = 0
        self.max_steps = 0

        self.n_classes: int = 0
        self.label_to_class_index: dict = {}
        self.class_index_to_label: dict = {}
        self.critical_classes: list[str] = []
        self.critical_class_mask: np.array = []

    @abstractmethod
    def _init_data_loader(self, *args, **kwargs) -> AbstractDataLoader:
        pass

    def _get_front(self, file_path):
        file_path = pathlib.Path(file_path)
        front_file_path = os.path.join(file_path.parent.resolve(), "pareto_frontiers", f"{file_path.stem}.csv")
        if os.path.isfile(front_file_path):
            self.front = pd.read_csv(front_file_path, index_col=0).values
        else:
            self.front = None

    def observe(self):
        return StepResponse(
            next_state=self.state,
            label=self.state_labels,
            terminal=self.terminal,
        )

    @abstractmethod
    def step(self, *args, **kwargs) -> StepResponse:
        """

        Parameters
        ----------
        action

        Returns
        -------
        StepResponse: {Next state, Reward, Terminal, Info}
        """
        pass

    @abstractmethod
    def reset(self, *args, **kwargs):
        return self.state

    def _make_critical_class_mask(self):
        critical_class_mask = np.zeros(self.n_classes)
        if len(self.critical_classes):
            critical_class_indices = [self.label_to_class_index[label] for label in self.critical_classes]

            critical_class_mask[critical_class_indices] = True

        return critical_class_mask.tolist()

    @staticmethod
    def get_critical_classes(config):
        critical_classes = config.get("critical_classes", [])

        if critical_classes is not None:
            critical_classes = [str(label) for label in critical_classes]
        else:
            critical_classes = []

        return critical_classes

    def extend_agent_config(self, config: dict) -> dict:
        """
        Responsible for copying the information from the environment to the config,
        which is also needed for the model initialization.

        Parameters
        ----------
        config

        Returns
        -------

        """
        dynamic_attributes = [
            "label_to_class_index",
            "class_index_to_label",
            "critical_classes",
            "critical_class_mask",
        ]

        for attribute in dynamic_attributes:
            config["agent_config"][attribute] = getattr(self, attribute)

        return config

    def get_label(self, *args, **kwargs):
        if self.run_mode != RunModes.INFERENCE.value:
            return self.state_labels
