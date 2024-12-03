import logging
from typing import TYPE_CHECKING, Tuple

import numpy as np
from tqdm import tqdm

from optilearn.configs.constants import RunModes
from optilearn.environments import AbstractEnv
from optilearn.utils.utils import get_device, set_seed_everywhere

if TYPE_CHECKING:
    from optilearn.meta_constructor import MetaConstructor


class Experiment:
    """Class representing an experiment for multi-objective optimization classification."""

    def __init__(
        self,
        config,
        meta_constructor: "MetaConstructor",
        log: bool,
        use_gpu_acceleration: bool = True,
    ):
        """
        Initialize the Experiment object.

        Args:
            config (dict): Configuration parameters for the experiment.
            meta_constructor: Meta constructor object for creating other objects.
            log: Log object for logging information.
            use_gpu_acceleration (bool): Flag indicating whether to use GPU acceleration.

        """
        set_seed_everywhere(config["seed"])
        self.config = config
        self.log = log
        self.device = get_device(use_gpu_acceleration)

        self.env, self.env_eval = self._init_envs(config, meta_constructor)

        config = self.env.extend_agent_config(config)
        self.agent = meta_constructor.agent(
            config["agent_config"],
            model=meta_constructor.model,
            u_func=meta_constructor.u_func,
            loss_criterion=meta_constructor.loss,
            device=self.device,
        )

        self.vis = meta_constructor.visualization(n_obj=config["agent_config"]["p_dim"])
        self.nl = meta_constructor.neptune_logger()
        self.evaluation = meta_constructor.evaluation(
            self.agent,
            config["agent_config"]["p_dim"],
            self.env_eval,
        )

        steps_per_episode = self.env.max_steps
        self.eval_freq = steps_per_episode * config["eval_freq"]
        self.training_steps = steps_per_episode * config["training_steps"]

    def _init_envs(self, config: dict, meta_constructor: "MetaConstructor") -> Tuple[AbstractEnv, AbstractEnv]:
        """
        Initialize the base environment and evaluation environment.

        Args:
            config (dict): Configuration parameters for the experiment.
            meta_constructor: experiment's meta constructor.

        Returns:
            tuple: A tuple containing the base environment and evaluation environment.

        """
        logging.info("Initializing base environment.")
        env = meta_constructor.env(
            config["env_config"],
            data_loader=meta_constructor.training_data_loader,
            transformation_pipeline=meta_constructor.training_transformation_pipeline,
            run_mode=RunModes.TRAINING,
            device=self.device,
        )

        if meta_constructor.eval_transformation_pipeline is None:
            eval_transformation_pipeline = env.loader.transformation_pipeline
        else:
            eval_transformation_pipeline = meta_constructor.eval_transformation_pipeline

        logging.info("Initializing evaluation environment.")
        env_eval = meta_constructor.env(
            config["env_config"],
            data_loader=meta_constructor.eval_data_loader,
            transformation_pipeline=eval_transformation_pipeline,
            run_mode=RunModes.EVAL,
            device=self.device,
        )

        return env, env_eval

    def _log_step(self, loss_1, loss_2, loss_3, training_rewards):
        """
        Log the step metrics.

        Args:
            loss_1: Loss 1 value.
            loss_2: Loss 2 value.
            loss_3: Loss 3 value.
            training_rewards: Training labels.

        """
        if len(training_rewards) > 0:
            self.nl.log_metric(
                metric_value=np.mean(training_rewards),
                metric_name="training_reward",
                mode="metrics",
            )
        self.nl.log_metric(metric_value=loss_1, metric_name="loss_1", mode="metrics")
        self.nl.log_metric(metric_value=loss_2, metric_name="training_accuracy", mode="metrics")
        self.nl.log_metric(metric_value=loss_3, metric_name="loss_3", mode="metrics")
        if hasattr(self.agent.model, "alpha"):
            self.nl.log_metric(
                metric_value=self.agent.model.alpha(),
                metric_name="alpha",
                mode="metrics",
            )
        if hasattr(self.agent.model, "target_entropy"):
            self.nl.log_metric(
                metric_value=self.agent.model.target_entropy(),
                metric_name="target_entropy",
                mode="metrics",
            )

    def _do_evaluation(self, step: int):
        """
        Perform evaluation and log the evaluation metrics.

        Args:
            step (int): The current step.

        """
        self.evaluation.run(n_prefs=self.config["n_prefs_eval"])
        # self.nl.log_metric(
        #     metric_value=self.evaluation.total_rewards.mean(),
        #     metric_name="total_reward",
        #     mode="metrics",
        #     step=step,
        # )
        self.nl.log_metric(
            metric_value=self.evaluation.accuracy,
            metric_name="accuracy",
            mode="metrics",
            step=step,
        )
        if self.agent.is_moo:
            for class_index, class_precision in enumerate(self.evaluation.binary_precision):
                self.nl.log_metric(
                    metric_value=class_precision,
                    metric_name=f"binary_precision_class{class_index}",
                    mode="metrics",
                    step=step,
                )

            for class_index, class_recall in enumerate(self.evaluation.binary_recall):
                self.nl.log_metric(
                    metric_value=class_recall,
                    metric_name=f"binary_recall_class{class_index}",
                    mode="metrics",
                    step=step,
                )

        self._log_figures(step)
        self._log_results(step)

        if self.config["log_model"]:
            self.nl._log_model_as_artifact("last", self.evaluation.models["last"]["model"])

    def _log_results(self, step):
        """
        Log the experiment results.

        Args:
            step (int): The current step.

        """
        pass

    def _log_figures(self, step, last=False):
        """
        Log the experiment figures.

        Args:
            step (int): The current step.
            last (bool): Flag indicating whether it is the last figure.

        """
        log_fun = self.nl.log_fig if not last else self.nl.upload_fig
        if self.evaluation.agent.is_moo:
            log_fun(
                self.vis.gen_xy_plot(
                    self.evaluation.eval_prefs,
                    self.evaluation.total_rewards,
                    step,
                    "Rewards_Pref",
                ),
                f"pareto_xy",
                "plots",
            )
        if self.evaluation.is_img_data:

            if len(self.evaluation.top_10_barely_wrong):
                log_fun(
                    self.vis.gen_plot_top_x(self.evaluation.top_10_barely_wrong, self.evaluation.observations),
                    f"top_10_barely_wrong",
                    "image_plots",
                )

            if len(self.evaluation.top_10_barely_right):
                log_fun(
                    self.vis.gen_plot_top_x(self.evaluation.top_10_barely_right, self.evaluation.observations),
                    f"top_10_barely_right",
                    "image_plots",
                )

            if len(self.evaluation.top_10_totally_wrong):
                log_fun(
                    self.vis.gen_plot_top_x(self.evaluation.top_10_totally_wrong, self.evaluation.observations),
                    f"top_10_totally_wrong",
                    "image_plots",
                )

            if len(self.evaluation.top_10_totally_wrong):
                log_fun(
                    self.vis.gen_plot_bar_category(self.evaluation.combined_metrics),
                    f"combined_metrics",
                    "bar_plots",
                )

    def run(self, tags):
        """
        Runs the training process.

        Args:
            tags (list): List of tags to associate with the run.

        Returns:
            tuple: A tuple containing two lists - `losses` and `performance`.
                - `losses` (list): List of loss values during training.
                - `performance` (list): List of performance metrics during training.
        """
        if self.log:
            self.nl.start(run_name=self.config["name"], tags=tags)
            self.nl.log_config(self.config)

        losses = []
        performance = []
        logging.info("Beginning training steps.")
        for step in tqdm(range(self.training_steps)):
            training_rewards = []
            loss_1, loss_2, loss_3 = self.agent.train(env=self.env, n_prefs=self.config["n_prefs_train"])
            losses.append(loss_1)
            performance.append(self.agent.performance_metric)

            if self.log:
                self._log_step(loss_1, loss_2, loss_3, training_rewards)
                if step == 0 or (step + 1) % self.eval_freq == 0:
                    self.evaluation.models["last"]["model"] = self.agent.model.state_dict()
                    self._do_evaluation(step + 1)
                if (step + 1) % (10 * self.eval_freq) == 0 and self.config["log_model"]:
                    self.nl.log_models(self.evaluation.models, self.config)
        self.evaluation.models["last"]["model"] = self.agent.model.state_dict()

        if self.log:
            self._log_figures(self.training_steps, last=True)
            if self.config["log_model"]:
                self.nl.log_models(self.evaluation.models, self.config)
                self.nl.log_model(
                    self.agent.model,
                    self.config,
                    self.env_eval.loader.transformation_pipeline,
                )
            self.nl.stop()

        return losses, performance
