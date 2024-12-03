import os
from time import time

from optilearn.utils.utils import load_yml
from optilearn import configs

def get_configs_path() -> str:
    """
    Get the path to the configuration files.

    Returns:
        str: The path to the configuration files.
    """
    return os.path.dirname(configs.__file__)

def setup(base_config_path: str, env_config_path: str = None, model_config_path: str = None, args=None):
    """
    Set up the configuration.

    Args:
        base_config_path : str
            Path to the main "configs" directory
        env_config_path : str, default = None
            if not None, env config will be loaded directly from here
        model_config_path : str, default = None
            if not None, model config will be loaded directly from here

        args: The command line arguments.

    Returns:
        tuple: A tuple containing the main configuration, tags, and iterations.

    """

    config_file_path = os.path.join(base_config_path, "config.yml")

    main_config = load_yml(config_file_path)
    main_config["config_path"] = config_file_path

    if args is not None:
        for key, value in vars(args).items():
            main_config[key] = value

    if env_config_path is None:
        env_config_path = os.path.join(
            base_config_path,
            "environments",
            main_config["env_config_file_name"],
        )

    if model_config_path is None:
        model_config_path = os.path.join(
            base_config_path,
            "model",
            main_config["model_config_file_name"],
        )

    env_config = load_yml(env_config_path)
    model_config = load_yml(model_config_path)

    class_subset = env_config.get("class_subset", None)
    if class_subset is not None:
        assert isinstance(class_subset, list), "<class_subset> is neither none nor a list"
        assert env_config["dim_out"] == len(set(class_subset)), "<dim_out> must match number of unique classes"
    if not main_config["moo"]:
        main_config["n_prefs_eval"] = 1
        main_config["n_prefs_train"] = 1
        env_config["n_obj"] = 0

    main_config["env_config"] = env_config

    main_config["agent_config"]["model_config"] = model_config
    main_config["agent_config"]["torch_interface"] = model_config.get("torch_interface", False)
    main_config["env_config"]["torch_interface"] = model_config.get("torch_interface", False)

    main_config["agent_config"]["p_dim"] = env_config["n_obj"]
    main_config["agent_config"]["a_dim"] = env_config["dim_out"]
    main_config["agent_config"]["s_dim"] = env_config["dim_state"]
    main_config["agent_config"]["critical_classes"] = env_config.get("critical_classes", None)

    main_config["constructor_keys"] = {
        "env_name": main_config["env_config"]["name"],
        "model_type": model_config["model_type"],
        "u_func_name": main_config["agent_config"]["u_func"],
        "loss_type": main_config["agent_config"]["loss_type"],
        "training_data_loader": main_config["env_config"]["training_data_loader"],
        "eval_data_loader": main_config["env_config"]["eval_data_loader"],
        "training_transformation_pipeline": main_config["env_config"]["data_loader_config"].get(
            "training_transformation_pipeline", "default"
        ),
        "eval_transformation_pipeline": main_config["env_config"]["data_loader_config"].get(
            "eval_transformation_pipeline", "default"
        ),
    }

    main_config["name"] = f'{env_config["name"]}_{model_config["model_type"]}_{main_config["agent_config"]["u_func"]}'

    tags = [model_config["model_type"], env_config["name"]]

    if main_config["name"] is not None:
        tags += [main_config["name"]]

    if "img_size" in main_config["env_config"]["data_loader_config"]:
        main_config["agent_config"]["model_config"]["input_size"] = main_config["env_config"]["data_loader_config"][
            "img_size"
        ]
    return main_config, tags, main_config.pop("iterations")


def run_exp(meta_constructor, config, tags, iterations=1):
    """
    Run the experiment.

    Args:
        meta_constructor: The meta constructor object.
        config (dict): The configuration dictionary.
        tags (list): The list of tags.
        iterations (int, optional): The number of iterations. Defaults to 1.

    """
    log = config["log"]
    use_gpu_acceleration = config["use_gpu_acceleration"]
    if iterations == 1 or not log:
        exp = meta_constructor.experiment(config, meta_constructor, log, use_gpu_acceleration)
        exp.run(tags)
    else:
        config["exp_id"] = int(time())
        for i in range(iterations):
            print("Iteration:", i + 1, "/", iterations)
            config["seed"] = i
            exp = meta_constructor.experiment(config, meta_constructor, log, use_gpu_acceleration)
            exp.run(tags)
