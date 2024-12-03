import os
from typing import Tuple

from optilearn.meta_constructor import MetaConstructor
from optilearn.utils.experiment_utils import run_exp, setup as base_setup
from object_maps import ObjectMap


def setup() -> Tuple:
    """
    Sets up configurations for training.

    Returns:
        main_config: The main configuration dictionary.
        tags: The tags for the training.
        iterations: The number of training iterations.
    """

    base_configs_path = os.path.join(os.path.dirname(__file__), "configs")
    env_config_path = os.path.join(base_configs_path, "environment.yml")
    model_config_path = os.path.join(base_configs_path, "model.yml")

    return base_setup(
        base_config_path=base_configs_path,
        env_config_path=env_config_path,
        model_config_path=model_config_path,
    )


if __name__ == "__main__":
    config, tags, iterations = setup()
    meta_constructor = MetaConstructor.from_config(config["constructor_keys"], ObjectMap)
    run_exp(meta_constructor, config, tags, iterations=iterations)
