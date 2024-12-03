import os
from typing import ClassVar, Dict

import joblib
import neptune
import torch

from optilearn.utils.utils import load_yml
import optilearn

MODEL_VERSION = "OP-IMACLADEF-8"
LOCAL_REGISTRY = os.path.join(os.path.dirname(os.path.dirname(optilearn.__file__)), "model_registry")
os.makedirs(LOCAL_REGISTRY, exist_ok=True)


class ModelRegistry:
    """
    ModelRegistry is responsible for handling model artifacts, which have been logged to
    the neptune model-registry.
    """

    MODEL_FILE_NAME: ClassVar[str] = "sk_like_model"
    TRANSFORMER_FILENAME: ClassVar[str] = "transforms"
    CONFIG_FILENAME: ClassVar[str] = "config.yml"

    MODEL_KEY: ClassVar[str] = "model"
    TRANSFORMER_KEY: ClassVar[str] = "transformation_pipeline"
    CONFIG_KEY: ClassVar[str] = "config"

    @classmethod
    def fetch_model_objects(cls, model_version: str = MODEL_VERSION, local_registry: str = LOCAL_REGISTRY) -> None:
        """
        Downloads the model and transformer objects of version <model_version> from the
        neptune-registry for Fetch, to the <local_registry>/<model_version>/ directory.

        Will Raise a FileNotFoundError if version <model_version> is not found in the
        registry or if the corresponding artifacts are not logged.

        """

        if cls.is_downloaded(model_version, local_registry):
            print(f"Model Version: {model_version} already downloaded.")
            return

        neptune_model = neptune.init_model_version(with_id=model_version, project="spryfox/OptiLearn", mode="read-only")

        if not neptune_model.exists("model"):
            raise FileNotFoundError(f"Model Version: {model_version} does not exist.")

        artifacts_path = cls.get_local_artifacts_paths(model_version, local_registry)

        neptune_model[f"model/{cls.MODEL_FILE_NAME}"].download(artifacts_path[cls.MODEL_KEY])
        neptune_model[f"model/{cls.CONFIG_KEY}"].download(artifacts_path[cls.CONFIG_KEY])

        if neptune_model.exists(f"model/{cls.TRANSFORMER_FILENAME}"):
            neptune_model[f"model/{cls.TRANSFORMER_FILENAME}"].download(artifacts_path[cls.TRANSFORMER_KEY])

        neptune_model.stop()

    @classmethod
    def get_local_artifacts_paths(cls, model_version: str, local_registry: str = LOCAL_REGISTRY) -> Dict:
        """
        Creates <local_registry>/<model_version>/ if it doesn't exist and returns the
        absolute paths of  model and config objects as a dict

        """
        model_dir = os.path.join(local_registry, model_version)
        os.makedirs(model_dir, exist_ok=True)

        return {
            cls.MODEL_KEY: os.path.join(model_dir, cls.MODEL_FILE_NAME),
            cls.CONFIG_KEY: os.path.join(model_dir, cls.CONFIG_FILENAME),
            cls.TRANSFORMER_KEY: os.path.join(model_dir, cls.TRANSFORMER_FILENAME),
        }

    @classmethod
    def is_downloaded(cls, model_version: str, local_registry: str = LOCAL_REGISTRY):
        """
        Returns a boolean flag indicating whether the corresponding model object
        exist in <model_registry>/<model_version>

        """
        model_dir = os.path.join(local_registry, model_version)
        downloaded: bool = False

        if os.path.exists(model_dir):
            artifacts_path: Dict = cls.get_local_artifacts_paths(model_version, local_registry)

            if os.path.exists(artifacts_path[cls.MODEL_KEY]):
                downloaded = True

        return downloaded

    @classmethod
    def load_artifacts(
        cls, model_version: str = MODEL_VERSION, device: str = "cpu", local_registry: str = LOCAL_REGISTRY
    ) -> Dict:
        artifacts_path = cls.get_local_artifacts_paths(model_version, local_registry)

        if not cls.is_downloaded(model_version, local_registry):
            cls.fetch_model_objects(model_version, local_registry)

        try:
            model = torch.jit.load(artifacts_path[cls.MODEL_KEY], map_location=device)
        except RuntimeError:
            model = joblib.load(artifacts_path[cls.MODEL_KEY])

        transformer = None
        if os.path.exists(artifacts_path[cls.TRANSFORMER_KEY]):
            try:
                transformer = torch.jit.load(artifacts_path[cls.TRANSFORMER_KEY], map_location=device)
            except RuntimeError:
                transformer = joblib.load(artifacts_path[cls.TRANSFORMER_KEY])

        return {
            cls.MODEL_KEY: model,
            cls.CONFIG_KEY: load_yml(artifacts_path[cls.CONFIG_KEY]),
            cls.TRANSFORMER_KEY: transformer,
        }


if __name__ == "__main__":
    model_artifacts = ModelRegistry.load_artifacts(model_version="OP-TABCLALGB-32")
