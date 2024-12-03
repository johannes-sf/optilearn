from typing import Type, TypeVar, Union
import pandas
import vaex
from numpy import array
from typing import TypeVar, Union

from optilearn.environments.data_loaders.transformations import AbstractTransformationPipeline

NpLikeArray = TypeVar("NpLikeArray", bound=Union[array, pandas.DataFrame, pandas.Series])
PandasArray = TypeVar("PandasArray", bound=Union[pandas.DataFrame, pandas.Series])

TransformationPipelineType = TypeVar("TransformationPipelineType", bound=AbstractTransformationPipeline)
TransformationPipeline = Union[TransformationPipelineType, Type[TransformationPipelineType], None]

DataFrameLike = TypeVar("DataFrameLike", bound=Union[vaex.DataFrame, pandas.DataFrame])