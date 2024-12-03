import io
import os
import tempfile
from time import time

import neptune as neptune
import plotly
import toml
import torch
from PIL import Image
from matplotlib import pyplot as plt
from neptune.types import File

from optilearn.configs.constants import *
from optilearn.configs.strings import *
from optilearn.environments.data_loaders.transformations import AbstractTransformationPipeline
from optilearn.utils.utils import load_yml, save_to_yml


class NeptuneLogger:
    """
    NeptuneLogger is a class for logging experiment metrics, configurations, and artifacts to Neptune.

    Attributes:
        run: The Neptune run object.
        init_time: The initialization time of the NeptuneLogger object.

    Methods:
        log_params: Logs the experiment parameters.
        log_config: Logs the experiment configuration.
        log_metric: Logs a metric value.
        log_fig: Logs a figure.
        upload_config: Uploads the experiment configuration as an artifact.
        upload_fig: Uploads a figure as an artifact.
        _log_model_as_artifact: Logs a model as an artifact.
        log_model: Logs a model and its configuration.
        _get_model_version: Gets the model version from Neptune.
        _log_model_metrics: Logs the model metrics.
        log_models: Logs multiple models.
        log_dataframe: Logs a pandas DataFrame.
        fetch_values: Fetches the values of a metric or artifact.
        download_metrics: Downloads the metrics as a CSV file.
        download_artifact: Downloads an artifact.
        download_model: Downloads a model and its configuration.
        start: Starts a Neptune run.
        stop: Stops the Neptune run.
    """

    def __init__(self):
        self.run = None
        self.init_time = int(time())

    def log_params(self, params):
        """
        Logs the experiment parameters.

        Args:
            params: A dictionary containing the experiment parameters.
        """
        self.run[PARAMETERS] = params

    def log_config(self, config):
        """
        Logs the experiment configuration.

        Args:
            config: A dictionary containing the experiment configuration.
        """
        self.run[CONFIG] = config
        self.upload_config(config)

    def log_metric(self, metric_value, metric_name, mode, step=None):
        """
        Logs a metric value.

        Args:
            metric_value: The value of the metric.
            metric_name: The name of the metric.
            mode: The mode of the metric (e.g., "train", "val").
            step: The step number (optional).
        """
        self.run[f"{mode}/{metric_name}"].log(metric_value, step=step)

    def log_fig(self, fig, fig_name, mode, step=None):
        """
        Logs a figure.

        Args:
            fig: The figure object.
            fig_name: The name of the figure.
            mode: The mode of the figure (e.g., "train", "val").
            step: The step number (optional).
        """
        if isinstance(fig, plotly.graph_objs.Figure):
            fig = Image.open(io.BytesIO(fig.to_image()))
        self.run[f"{mode}/{fig_name}"].log(fig, step=step)
        plt.clf()
        plt.close("all")

    def upload_config(self, config, model_version=None):
        """
        Uploads the experiment configuration as an artifact.

        Args:
            config: A dictionary containing the experiment configuration.
            model_version: The model version (optional).
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = save_to_yml(config, CONFIG_FILE, tmp_dir)
            if model_version is None:
                self.run[f"{MODEL}/{CONFIG}"].upload(config_path, wait=True)
            else:
                model_version[f"{MODEL}/{CONFIG}"].upload(config_path, wait=True)

    def upload_fig(self, fig, fig_name, mode):
        """
        Uploads a figure as an artifact.

        Args:
            fig: The figure object.
            fig_name: The name of the figure.
            mode: The mode of the figure (e.g., "train", "val").
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            if isinstance(fig, plt.Figure):
                img_path = os.path.join(tmp_dir, f"{fig_name}.png")
                plt.savefig(img_path)
            else:
                img_path = os.path.join(tmp_dir, f"{fig_name}.html")
                fig.write_html(img_path, include_mathjax="cdn", include_plotlyjs="cdn")
            self.run[f"{mode}/{fig_name}_ul"].upload(img_path, wait=True)
            plt.clf()
            plt.close("all")

    def _log_model_as_artifact(self, model_name, model):
        """
        Logs a model as an artifact.

        Args:
            model_name: The name of the model.
            model: The model object.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, f"{model_name}.mod")
            torch.save(model, model_path)
            self.run[f"{MODEL}/{model_name}"].upload(model_path, wait=True)

    def log_model(self, model, config, transforms=None):
        """
        Logs a model and its configuration.

        Args:
            model: The model object.
            config: A dictionary containing the model configuration.
            transforms: The transforms object (optional).
        """
        name = config["name"]
        key = "".join([i[: min(len(i), 3)] for i in config["name"].split("_")[:3]]).upper()
        model_version = self._get_model_version(key, name)
        self._log_model_metrics(model_version, model, config)

        if config["agent_config"]["torch_interface"]:
            self._log_torch_model(model_version, model, transforms)
        else:
            self._log_sk_like_model(model_version, model, transforms)

    @staticmethod
    def _log_sk_like_model(model_version, model, transforms: AbstractTransformationPipeline = None):
        model_version[f"{MODEL}/sk_like_model"].upload(File.as_pickle(model))
        if transforms is not None:
            model_version[f"{MODEL}/transforms"].upload(File.as_pickle(transforms))

    @staticmethod
    def _log_torch_model(model_version, model, transforms):
        scripted_model = torch.jit.script(model)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, f"model.pt")
            scripted_model.save(model_path)
            model_version[f"{MODEL}/torch_script"].upload(model_path, wait=True)

            if transforms is not None:
                transforms_path = os.path.join(tmp_dir, f"transforms.pt")
                scripted_transforms = torch.jit.script(transforms)
                scripted_transforms.save(transforms_path)
                model_version[f"{MODEL}/transforms_script"].upload(transforms_path, wait=True)
        # scripted_model = torch.jit.script(model)
        # model_version[f"{MODEL}/torch_script_model"].upload(File.as_pickle(scripted_model))
        # if transforms is not None:
        #     scripted_transforms = torch.jit.script(transforms)
        #     model_version[f"{MODEL}/transforms_script"].upload(File.as_pickle(scripted_transforms))

    def _get_model_version(self, key, name):
        """
        Gets the model version from Neptune.

        Args:
            key: The model key.
            name: The model name.

        Returns:
            The model version object.
        """
        try:
            model_version = neptune.init_model(name=name, key=key, project=PROJECT)
        except neptune.exceptions.NeptuneModelKeyAlreadyExistsError as error:
            print("model already exists, caught error : NeptuneModelKeyAlreadyExistsError ")
            print("Creating new model...")
        sys_id = self.run["sys/id"].fetch().split("-")[0]
        model_version = neptune.init_model_version(name=name, model=sys_id + "-" + key, project=PROJECT)
        return model_version

    def _log_model_metrics(self, model_version, model, config):
        """
        Logs the model metrics.

        Args:
            model_version: The model version object.
            model: The model object.
            config: A dictionary containing the model configuration.
        """
        model_version["run/id"] = self.run["sys/id"].fetch()
        model_version["run/url"] = self.run.get_url()
        model_version["version"] = toml.load("../pyproject.toml")["tool"]["poetry"]["version"]
        # model_version["run/val_class_acc"] = self.run["metric/aggr/val_class_acc"].fetch_last()
        # model_version["run/val_sample_acc"] = self.run["metric/aggr/val_sample_acc"].fetch_last()
        # model_version[f"{MODEL}/classes"] = model.classes
        # model_version[f"{MODEL}/img_size"] = model.img_size
        # model_version[f"{MODEL}/n_num_features"] = model.n_num_features
        if config is not None:
            self.upload_config(config, model_version)
            # model_version[f"{MODEL}/config"] = config

    def log_models(self, models, config):
        """
        Logs multiple models.

        Args:
            models: A dictionary containing the models.
            config: A dictionary containing the model configuration.
        """
        # self.upload_config(config)
        for model_name, model in models.items():
            self._log_model_as_artifact(model_name, model["model"])

    def log_dataframe(self, df, df_name, mode):
        """
        Logs a pandas DataFrame.

        Args:
            df: The pandas DataFrame.
            df_name: The name of the DataFrame.
            mode: The mode of the DataFrame (e.g., "train", "val").
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, f"{df_name}.csv")
            df.to_csv(csv_path)
            self.run[f"{mode}/{df_name}"].upload(csv_path, wait=True)

    def fetch_values(self, folder, name):
        """
        Fetches the values of a metric or artifact.

        Args:
            folder: The folder containing the metric or artifact.
            name: The name of the metric or artifact.

        Returns:
            The fetched values.
        """
        return self.run[f"{folder}/{name.split('.')[0]}"].fetch_values()

    def download_metrics(self, folder, name, save_path):
        """
        Downloads the metrics as a CSV file.

        Args:
            folder: The folder containing the metrics.
            name: The name of the metrics.
            save_path: The path to save the downloaded file.

        Returns:
            The file path of the downloaded CSV file.
        """
        file_path = os.path.join(save_path, f"{name}.csv")
        df = self.fetch_values(folder, name)
        df.to_csv(file_path)
        return file_path

    def download_artifact(self, folder, name, save_path) -> str:
        """
        Downloads an artifact.

        Args:
            folder: The folder containing the artifact.
            name: The name of the artifact.
            save_path: The path to save the downloaded file.

        Returns:
            The file path of the downloaded artifact.
        """
        file_path = os.path.join(save_path, name)
        self.run[f"{folder}/{name.split('.')[0]}"].download(save_path)
        return file_path

    def download_model(self, algorithm):
        """
        Downloads a model and its configuration.

        Args:
            algorithm: The name of the model.

        Returns:
            The downloaded model and its configuration.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = self.download_artifact("model", f"{algorithm}.mod", tmp_dir)
            config_path = self.download_artifact("model", f"{CONFIG}.yml", tmp_dir)
            model = torch.load(model_path)
            config = load_yml(config_path)
        return model, config

    def start(self, run_name, tags, run_id=None):
        """
        Starts a Neptune run.

        Args:
            run_name: The name of the run.
            tags: The tags for the run.
            run_id: The ID of the run (optional).
        """
        if run_id is not None:
            self.run = neptune.init_run(project=PROJECT, with_id=run_id)
        else:
            self.run = neptune.init_run(project=PROJECT, tags=tags, name=run_name)

    def stop(self):
        """
        Stops the Neptune run.
        """
        self.run.stop()
