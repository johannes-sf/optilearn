import logging
import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import vaex
from pandas import DataFrame, Series

from optilearn.configs.constants import LabelType
from optilearn.environments.data_loaders import AbstractDataLoader
from optilearn.environments.responses import DataLoaderResponse
from optilearn.utils.custom_types import DataFrameLike, TransformationPipeline
from optilearn.utils.dataset.metadata.dto import TrainingMetaData


@dataclass
class TabularDataLoaderResponse(DataLoaderResponse):
    """
    A response class for the TabularDataLoader containing the state and o optional state labels.

    Attributes:
        state (DataFrame): The state data.
        state_labels (Optional[Series]): The state labels, default is None.
    """

    state: DataFrame
    state_labels: Optional[Series] = None


class TabularDataLoader(AbstractDataLoader):
    def __init__(
        self,
        data_path: str,
        learning_mode: bool = True,
        columns: List[str] = None,
        index_column: str = None,
        label_column: str = "Label",
        label_type: LabelType = LabelType.NORMAL,
        shuffle: bool = True,
        batch_size: int = None,
        torch_interface: bool = True,
        nrows: int = None,
        transformation_pipeline: TransformationPipeline = None,
        oom=False,
        **kwargs,
    ):
        """
        Initializes the TabularDataLoader.

        Args:
            data_path (str): Path to the CSV file.
            learning_mode (bool): Indicates whether to load data for training or inference.
            columns (List[str]): List of columns to load. If None, all columns in the CSV will be loaded.
            index_column (str): Column to use as the index of the resulting DataFrame.
            label_column (str): Column to use as labels/state for training.
            label_type (LabelType): How to handle the labels.
            transformation_pipeline (Callable): A callable for feature engineering. When learning_mode is true,
                                                this object is expected to be pre-initialized
            shuffle (bool): Currently not implemented for tabular data.
            batch_size (int): Number of rows per batch of DataFrame.
            nrows (int): Number of rows of the table to read (for debugging).
            **kwargs: Additional keyword arguments.
        """
        AbstractDataLoader.__init__(
            self,
            learning_mode=learning_mode,
            shuffle=shuffle,
            batch_size=batch_size,
            torch_interface=torch_interface,
            **kwargs,
        )

        self.nrows = nrows
        self.data_path = data_path
        self.columns = columns
        self.label_type = label_type

        self.index_column = index_column
        self.label_column = label_column

        self.learning_mode = learning_mode
        self.total_number_of_rows: int = -1
        self._state_columns = []
        self.label_encoding = None

        self.oom = oom

        self.transformation_pipeline = self._init_transformation_pipeline(transformation_pipeline)
        self.init_loader()

    def init_loader(self):
        self._set_iterator_properties()

        data = self.load_data()

        if self.learning_mode:
            self.transformation_pipeline.fit(data.drop(columns=self.label_column))

        if self.oom:
            from vaex.ml import OneHotEncoder

            self.label_encoding = OneHotEncoder(features=self.label_column)
            self.label_encoding.fit(data)

        else:
            from sklearn.preprocessing import OneHotEncoder

            self.label_encoding = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            self.label_encoding.fit(data[[self.label_column]])

        data = self.transformation_pipeline.transform(data)

        self._iterator = self.make_iterator(data)
        self._state_columns = self.get_state_columns()

        self.meta_data = self._make_meta_data()

    def _init_transformation_pipeline(self, transformation_pipeline):
        if self.learning_mode:
            pipeline = transformation_pipeline()
        else:
            pipeline = transformation_pipeline

        return pipeline

    def step(self) -> TabularDataLoaderResponse:
        """
        Fetches the next batch of data.

        Returns:
            TabularDataLoaderResponse: The response containing the state and state labels.
        """
        batch: DataFrameLike = next(self.iterator)

        self._current_iteration += 1

        if self.current_iteration == self.total_iterations:
            self.terminal = True

        state = batch[self.state_columns].drop(columns=self.label_column).values
        state_labels = self.label_encoding.transform(batch[[self.label_column]])

        if self.oom:
            # Vaex OneHotEncoder returns the original column after transform
            # Also returns a dataframe instead of a numpy array
            state_labels = state_labels.drop(self.label_column).values

        # ToDo: Handle nans before trying to convert to tensors

        return TabularDataLoaderResponse(
            state_labels=self.conform_to_interface(state_labels),
            state=self.conform_to_interface(state),
        )

        # try:
        #
        #     return TabularDataLoaderResponse(
        #         state_labels=self.conform_to_interface(pd.get_dummies(batch[self.label_column]).values),
        #         state=self.conform_to_interface(),
        #     )
        # except TypeError:
        #
        #     return TabularDataLoaderResponse(
        #         state_labels=self.conform_to_interface(
        #             self.label_encoding.transform(batch[[self.label_column]]).drop(self.label_column).values
        #         ),
        #         state=self.conform_to_interface(batch[self.state_columns].values),
        #     )

    def reset(self):
        """
        Resets the data iterator.
        """
        self.init_loader()

    def _set_iterator_properties(self):
        """
        Sets properties related to the data iterator, such as total number of rows and total iterations.
        """

        if self.nrows:
            self.total_number_of_rows = self.nrows
        else:
            self.total_number_of_rows = (
                int(subprocess.check_output(f"wc -l {self.data_path}", shell=True).split()[0]) - 1
            )
        if self.batch_size is not None:
            self._total_iterations = max(self.total_number_of_rows // self.batch_size, 1)
        else:
            self._total_iterations = 1

    def make_iterator(self, data):
        """
        Creates an iterator for reading the CSV file in chunks.

        Yields:
            DataFrame: A chunk of the CSV file after applying the transformation pipeline.
        """

        if self.batch_size is None:
            chunk_size = self.total_number_of_rows
        else:
            chunk_size = min(self.batch_size, self.total_number_of_rows)

        for start in range(0, len(data), chunk_size):
            end = min(start + chunk_size, len(data))
            yield data[start:end]

    def load_data(self) -> DataFrameLike:
        logging.info(f"Loading dataset: {os.path.basename(self.data_path)} as data frame.")
        logging.info(f"{self.total_number_of_rows} rows to read.")

        if not self.oom:
            data_iterator = pd.read_csv(
                self.data_path,
                usecols=self.columns,
                nrows=self.nrows,
                iterator=True,
                chunksize=100000,
                low_memory=False,
                index_col=self.index_column,
            )
            data: pd.DataFrame = pd.concat([chunk for chunk in data_iterator])
        else:
            data = vaex.open(self.data_path, convert=True)

        return data

    def get_state_columns(self):
        """
        Sets the state columns by reading the CSV file header.

        Returns:
            List[str]: List of state columns.
        """

        return self.transformation_pipeline.feature_names

    def _make_meta_data(self):
        """
        Creates metadata for the training data.

        Returns:
            TrainingMetaData: The metadata for the training data.
        """
        labels: List = pd.read_csv(self.data_path, usecols=[self.label_column]).squeeze().unique().tolist()

        label_to_class_index = {label: iterator for iterator, label in enumerate(sorted(labels))}
        class_index_to_label = {index: label for label, index in label_to_class_index.items()}

        return TrainingMetaData(
            label_to_class_index=label_to_class_index,
            class_index_to_label=class_index_to_label,
            labels=labels,
            n_labels=len(labels),
            batch_size=self.batch_size,
            total_iterations=self.total_iterations,
            total_samples=self.total_number_of_rows,
        )

    @property
    def state_columns(self):
        """
        Property to get the state columns.

        Returns:
            List[str]: List of state columns.
        """
        return self._state_columns
