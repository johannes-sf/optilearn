from dataclasses import dataclass
from typing import ClassVar, Type

from optilearn.agent import Agent
from optilearn.base_object import BaseDataClass, BaseObject
from optilearn.environments import (
    AbstractEnv,
    ClassificationEnv,
)
from optilearn.environments.data_loaders import (
    AbstractDataLoader,
    LearningImageLoader,
    SimpleImageLoader,
    TabularDataLoader,
)
from optilearn.environments.data_loaders.transformations import (
    AbstractTransformationPipeline,
)
from optilearn.evaluation.evaluation import Evaluation
from optilearn.evaluation.neptune_logger import NeptuneLogger
from optilearn.evaluation.visualization import Visualization
from optilearn.experiment import Experiment
from optilearn.models import (
    AbstractModel,
    ClassificationLGBM,
    ClassificationNN,
    ClassificationSK,
)
from optilearn.utils.loss_funcs import *
from optilearn.utils.u_funcs import *


class Commons(BaseDataClass):
    """
    A class to hold common components used in the classification system.
    """
    agent: Type[Agent] = Agent
    experiment: Type[Experiment] = Experiment
    evaluation: Type[Evaluation] = Evaluation
    neptune_logger: Type[NeptuneLogger] = NeptuneLogger
    visualization: Type[Visualization] = Visualization


class DataLoaders(BaseObject):
    """
    A class to hold different types of data loaders.
    """
    abstract_base_type: ClassVar = AbstractDataLoader

    simple: Type[AbstractDataLoader] = SimpleImageLoader
    contamination: Type[AbstractDataLoader] = LearningImageLoader
    tabular: Type[AbstractDataLoader] = TabularDataLoader


class Models(BaseObject):
    """
    A class to hold different types of models.
    """
    abstract_base_type: ClassVar = AbstractModel

    nn: Type[AbstractModel] = ClassificationNN
    sk: Type[AbstractModel] = ClassificationSK
    lgbm: Type[AbstractModel] = ClassificationLGBM


class LossFuncs(BaseObject):
    """
    A class to hold different types of loss functions.
    """
    abstract_base_type: ClassVar = AbstractLoss

    cross_entropy: Type[AbstractLoss] = CustomCrossEntropyLoss
    weighted_cross_entropy: Type[AbstractLoss] = WeightedCrossEntropyLoss
    weighted_classes: Type[AbstractLoss] = WeightedClassLoss
    binary_precision_recall: Type[AbstractLoss] = BinaryPrecisionRecallLoss
    naive_cost: Type[AbstractLoss] = NaiveCostLoss
    f_beta_loss: Type[AbstractLoss] = FBetaLoss


class UFuncs(BaseObject):
    """
    A class to hold different types of utility functions.
    """
    abstract_base_type: ClassVar = AbstractUFunc

    linear: Type[AbstractUFunc] = Linear
    square: Type[AbstractUFunc] = Square
    log: Type[AbstractUFunc] = Log


class Envs(BaseObject):
    """
    A class to hold different types of environments.
    """
    abstract_base_type = AbstractEnv

    classification: Type[AbstractUFunc] = ClassificationEnv
    tabular_classification: Type[AbstractUFunc] = ClassificationEnv


class TransformationPipelines(BaseObject):
    """
    A class to hold different types of transformation pipelines.
    """
    abstract_base_type: ClassVar = AbstractTransformationPipeline

    default: Type[AbstractTransformationPipeline] = AbstractTransformationPipeline


@dataclass
class ObjectMap(BaseDataClass):
    """
    A dataclass to map all the components used in the classification system.
    """
    commons: Commons = Commons()
    data_loaders: DataLoaders = DataLoaders()
    transformation_pipelines: TransformationPipelines = TransformationPipelines()
    models: Models = Models()
    loss_funcs: LossFuncs = LossFuncs()
    u_funcs: UFuncs = UFuncs()
    envs: Envs = Envs()
