from typing import Type, Union

from torch import Tensor

from optilearn.configs.constants import RunModes
from optilearn.environments import AbstractEnv
from optilearn.environments.data_loaders import AbstractDataLoader
from optilearn.environments.data_loaders.transformations import AbstractTransformationPipeline
from optilearn.environments.responses import DataLoaderResponse, StepResponse


class ClassificationEnv(AbstractEnv):
    def __init__(
        self,
        config: dict,
        data_loader: Type[AbstractDataLoader],
        transformation_pipeline: Type[AbstractTransformationPipeline],
        run_mode: RunModes = RunModes.TRAINING,
        device="cpu",
    ):
        super().__init__(config, run_mode=run_mode, device=device, transformation_pipeline=transformation_pipeline)

        self.loader = self._init_data_loader(
            data_loader=data_loader,
            transformation_pipeline=transformation_pipeline,
        )

        self.max_steps = self.loader.meta_data.total_iterations
        self.n_classes = self.loader.meta_data.n_labels
        self.label_to_class_index = self.loader.meta_data.label_to_class_index
        self.class_index_to_label = self.loader.meta_data.class_index_to_label
        self.critical_classes = self.get_critical_classes(config)
        self.critical_class_mask = self._make_critical_class_mask()

        self.max_steps = self.loader.meta_data.total_iterations

        self._set_state_variables(self.loader.step())

    def _init_data_loader(
        self,
        data_loader: Type[AbstractDataLoader],
        transformation_pipeline: Type[AbstractTransformationPipeline],
    ) -> AbstractDataLoader:

        data_path = self.config["data_loader_config"][f"{self.run_mode.value}_data_path"]

        return data_loader(
            data_path=data_path,
            learning_mode=self.run_mode == RunModes.TRAINING,
            device=self.device,
            torch_interface=self.torch_interface,
            transformation_pipeline=transformation_pipeline,
            **self.config["data_loader_config"],
        )

    def step(self, **kwargs) -> StepResponse:
        # labels = self.get_label(pred)

        if self.terminal:
            raise StopIteration("Data loader already been exhausted. Reset environment.")

        next_sample: DataLoaderResponse = self.loader.step()
        self.terminal = self.loader.terminal
        self._set_state_variables(next_sample)
        self.steps += 1

        return StepResponse(
            next_state=self.state,
            label=self.state_labels,
            terminal=self.terminal,
        )

    def get_label(self, pred: Tensor = None, **kwargs) -> Union[Tensor, None]:
        """
        Calculates reward for the given pred. Not applicable for RunModes.INFERENCE.

        For classification reward is calculated as (label - pred) or in other words,
        one hot class label vector minus predicted class probabilities.

        The one hot class label vector will be recreated in the ClassificationNN model
        by adding simply adding reward and pred vectors.

        This workflow is chosen to stay true to the definition of "reward".

        Parameters
        ----------
        pred
        kwargs

        Returns
        -------
        reward: reward Tensor

        """
        if self.run_mode != RunModes.INFERENCE.value:
            return self.state_labels

    def reset(self) -> None:
        """
        Resets the image loader to the first iteration. This can be called e.g. when
        the iterator of the image loader has exhausted.

        Returns
        -------

        """

        self.loader.reset()
        self.terminal = False
        self._set_state_variables(self.loader.step())
        self.steps = 0

    def _set_state_variables(self, sample: DataLoaderResponse):
        """
        Sets the correct state for the environment based on the sample
        Parameters
        ----------
        sample

        Returns
        -------

        """
        self.state = sample.state
        self.state_labels = sample.state_labels
        self.terminal = self.loader.terminal
