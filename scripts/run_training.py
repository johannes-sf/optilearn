import os
import sys

from optilearn.utils.experiment_utils import setup as base_setup

# Add the root directory of your project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import argparse
from optilearn.meta_constructor import MetaConstructor
from optilearn.object_maps import ObjectMap
from optilearn.utils.experiment_utils import run_exp, get_configs_path


def setup(configs_path: str = get_configs_path()):
    """
    Set up the configuration for training.

    Args:
        configs_path: path to configs

    Returns:
        main_config: The main configuration dictionary.
        tags: The tags for the training.
        iterations: The number of training iterations.
    """
    main_config, tags, iterations = base_setup(base_config_path=configs_path)

    # add specific arguments here:

    return main_config, tags, iterations


if __name__ == "__main__":
    config, tags, iterations = setup(get_configs_path())
    meta_constructor = MetaConstructor.from_config(config["constructor_keys"], ObjectMap)
    run_exp(meta_constructor, config, tags, iterations=iterations)
