from typing import Type

from optilearn.agent import Agent
from optilearn.base_object import BasePydanticModel
from optilearn.environments import AbstractEnv
from optilearn.environments.data_loaders import AbstractDataLoader
from optilearn.environments.data_loaders.transformations import AbstractTransformationPipeline
from optilearn.evaluation.evaluation import Evaluation
from optilearn.evaluation.neptune_logger import NeptuneLogger
from optilearn.evaluation.visualization import Visualization
from optilearn.experiment import Experiment
from optilearn.models.abstract_model import AbstractModel
from optilearn.object_maps import ObjectMap
from optilearn.utils.custom_types import TransformationPipeline
from optilearn.utils.loss_funcs import AbstractLoss
from optilearn.utils.u_funcs import AbstractUFunc


class MetaConstructor(BasePydanticModel):
    """
    A class to construct meta objects for the classification system.
    """
    env: Type[AbstractEnv]
    experiment: Type[Experiment]
    agent: Type[Agent]
    evaluation: Type[Evaluation]
    neptune_logger: Type[NeptuneLogger]
    visualization: Type[Visualization]

    model: Type[AbstractModel]
    loss: Type[AbstractLoss]
    u_func: Type[AbstractUFunc]
    training_data_loader: Type[AbstractDataLoader]
    eval_data_loader: Type[AbstractDataLoader]

    training_transformation_pipeline: TransformationPipeline
    eval_transformation_pipeline: TransformationPipeline

    @classmethod
    def from_config(cls, config: dict, objects_map: Type[ObjectMap] = ObjectMap):
        """
        Create a MetaConstructor instance from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing the necessary parameters.
            objects_map (Type[ObjectMap], optional): Object map to retrieve components. Defaults to ObjectMap.

        Returns:
            MetaConstructor: An instance of MetaConstructor.
        """
        if config.get("eval_transformation_pipeline", None) is None:
            eval_transformation_pipeline = None
        else:
            eval_transformation_pipeline = getattr(
                objects_map.transformation_pipelines, config["eval_transformation_pipeline"]
            )

        return cls(
            agent=objects_map.commons.agent,
            experiment=objects_map.commons.experiment,
            evaluation=objects_map.commons.evaluation,
            neptune_logger=objects_map.commons.neptune_logger,
            visualization=objects_map.commons.visualization,
            env=getattr(objects_map.envs, config["env_name"]),
            model=getattr(objects_map.models, config["model_type"]),
            loss=getattr(objects_map.loss_funcs, config["loss_type"]),
            u_func=getattr(objects_map.u_funcs, config["u_func_name"]),
            training_data_loader=getattr(objects_map.data_loaders, config["training_data_loader"]),
            eval_data_loader=getattr(objects_map.data_loaders, config["eval_data_loader"]),
            training_transformation_pipeline=getattr(
                objects_map.transformation_pipelines, config["training_transformation_pipeline"]
            ),
            eval_transformation_pipeline=eval_transformation_pipeline,
        )
