from typing import Iterable

import numpy as np
import torch

from optilearn.inference.inferencer import AbstractInferencer


class Inferencer(AbstractInferencer):
    def __init__(
        self,
        model_version: str,
        device: str = "cpu",
        local_registry: str = "",
    ):
        super().__init__(model_version, device, local_registry)

    def infer(
        self,
        model_input: Iterable,
        preference: list[float] = None,
        top_k: int = None,
    ) -> dict[str:float]:
        """
        Run the model against the input.

        If preference is None, a uniformly distributed tensor will be fed to the model.

        ----------
        model_input
        preference
        top_k

        Returns
        -------
        Dictionary of class index: predicted probability

        """
        model_input = self.process_input(model_input)
        preference_tensor = self.process_preference(preference)

        model_output = self.model.forward(model_input, pref=preference_tensor)

        if top_k:
            probabilities, class_indices = torch.topk(model_output, k=top_k)
        else:
            probabilities = [model_output.max().unsqueeze(0)]
            class_indices = [model_output.argmax().unsqueeze(0)]

        output = {
            self.class_index_to_label[class_index.item()]: probability.item()
            for class_index, probability in list(zip(*class_indices, *probabilities))
        }

        return output

    def process_preference(self, preference: list[float] = None) -> torch.Tensor:

        if preference is None:
            # Create a uniform distribution over preferences
            preference = np.ones(self.p_dim) * [1 / self.p_dim]

        if not len(preference) == self.p_dim:
            raise ValueError(f"Preference must be of length {self.p_dim}.")

        # Un-squeeze in batch dimension
        return torch.tensor(preference, device=self.device, dtype=torch.float32).unsqueeze(0)

    def process_input(self, model_input: Iterable) -> torch.Tensor:
        if not isinstance(model_input, torch.Tensor):
            model_input = torch.tensor(model_input)

        if model_input.ndim == self.s_dim:
            # Un-squeeze in batch dimension
            model_input = model_input.unsqueeze(0)

        elif model_input.ndim - 1 == self.s_dim:
            pass
        else:
            raise ValueError(f"Invalid input for model: {self.model_version}")

        model_input = model_input.to(device=self.device)

        return model_input
