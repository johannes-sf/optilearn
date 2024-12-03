import os

# from feature_generator import FeatureGenerator
from optilearn.environments.data_loaders import TabularDataLoader
from optilearn.environments.data_loaders.transformations import AbstractTransformationPipeline


def test_tabular_data_loader(testing_data_dir):
    path = os.path.join(testing_data_dir, "tabular.csv")

    loader = TabularDataLoader(
        data_path=path,
        batch_size=None,
        index_column="PetId",
        label_column="Churn",
        nrows=1000,
        transformation_pipeline=AbstractTransformationPipeline,
        torch_interface=False,
    )

    terminal = False

    chunks = []

    while not terminal:
        step_response = loader.step()
        chunks.append(step_response.state)
        terminal = loader.terminal
