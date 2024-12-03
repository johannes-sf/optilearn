from abc import abstractmethod
from typing import Iterable

from optilearn.inference.model_registry import ModelRegistry


class AbstractInferencer:
    def __init__(
        self,
        model_version: str,
        device: str = "cpu",
        local_registry: str = "",
    ):
        registry_artifacts = ModelRegistry.load_artifacts(
            model_version=model_version, device=device, local_registry=local_registry
        )

        self.model_version = model_version
        self.local_registry = local_registry
        self.model = registry_artifacts["model"]
        self.transformation_pipeline = registry_artifacts["transformation_pipeline"]
        self.config = registry_artifacts["config"]
        self.class_index_to_label = self.config["agent_config"]["class_index_to_label"]
        self.p_dim = self.config["agent_config"]["p_dim"]
        self.s_dim = self.config["agent_config"]["s_dim"]

        self.device = device

    @abstractmethod
    def infer(self, model_input: Iterable, preference: list[float] = None, **kwargs) -> dict[str:float]:
        """
        Run the model against the input.

        If preference is None, a uniformly distributed vector will be fed to the model.

        ----------
        model_input
        preference
        top_k

        Returns
        -------
        Dictionary of class label: predicted probability

        """
        pass

    @abstractmethod
    def process_preference(self, preference: list[float] = None) -> Iterable:
        """
        Process the preference vector.

        e.g. normalize the preference vector to a unit vector or convert to tensor/numpy array, 
        depending on the model's input format.

        """
        pass

    @abstractmethod
    def process_input(self, model_input: Iterable) -> Iterable:
        """
        Process the model input.

        e.g. convert to tensor/numpy array, depending on the model's input format.

        """
        pass
