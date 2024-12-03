from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from optilearn.utils.data_frame_analyzer import DataFrameAnalyzer
from optilearn.utils.utils import get_eval_w, tensor_to_numpy


# from line_profiler import profile


class Evaluation:
    """
    A class to evaluate the performance of a multi-objective classification agent.

    Attributes:
        EPSILON (float): A small value to avoid division by zero.
        agent (object): The agent to be evaluated.
        n_obj (int): Number of objectives.
        env_eval (object): The evaluation environment.
        df_results (pd.DataFrame): DataFrame to store evaluation results.
        actions (list): List to store actions taken by the agent.
        probs (list): List to store probabilities predicted by the agent.
        labels (list): List to store true labels.
        labels_array (list): List to store the array of true labels.
        pref_cols (list): List to store preference column names.
        reward_cols (list): List to store reward column names.
        eval_prefs (int): Number of evaluation preferences.
        total_rewards (int): Total rewards accumulated.
        pref_loss (int): Preference loss.
        utility (int): Utility value.
        hypervolume (int): Hypervolume value.
        reward_opt (None): Optimal reward.
        models (dict): Dictionary to store the best models based on different metrics.
        u_func (function): Utility function.
        is_img_data (bool): Flag to check if the data is image data.
        accuracy (int): Accuracy of the model.
        weighted_accuracy (int): Weighted accuracy of the model.
        binary_precision (list): List to store binary precision values.
        binary_recall (list): List to store binary recall values.
        preference_array (np.array): Array to store preferences.
        observations (np.array): Array to store observations.
        top_10_barely_wrong (pd.DataFrame): DataFrame to store top 10 barely wrong predictions.
        top_10_barely_right (pd.DataFrame): DataFrame to store top 10 barely right predictions.
        top_10_totally_wrong (pd.DataFrame): DataFrame to store top 10 totally wrong predictions.
        combined_metrics (pd.DataFrame): DataFrame to store combined metrics.
    """

    EPSILON = 1e-5

    def __init__(self, agent, n_obj, env_eval, **kwargs):
        """
        Initializes the Evaluation class with the given agent, number of objectives, and evaluation environment.

        Args:
            agent (object): The agent to be evaluated.
            n_obj (int): Number of objectives.
            env_eval (object): The evaluation environment.
            kwargs (dict): Additional keyword arguments.
        """
        self.agent = agent
        self.df_results = pd.DataFrame()
        self.n_obj = n_obj
        self.env_eval = env_eval
        self.actions = []
        self.probs = []
        self.labels = []
        self.labels_array = []
        self.pref_cols = []
        self.reward_cols = []
        if n_obj <= 1:
            self.pref_cols.append("pref")
            self.reward_cols.append("reward")
        else:
            for obj in range(n_obj):
                self.pref_cols.append(f"pref_{obj}")
                self.reward_cols.append(f"reward_{obj}")
        self.eval_prefs = 0
        self.total_rewards = 0
        self.pref_loss = 0
        self.utility = 0
        self.hypervolume = 0
        self.reward_opt = None
        self.models = {
            "best_hv": {"value": -np.inf, "model": None},
            "best_utility": {"value": -np.inf, "model": None},
            "best_pref_loss": {"value": -np.inf, "model": None},
            "last": {"model": None},
        }
        self.u_func = self._init_ufunc()
        self.is_img_data = "img_size" in env_eval.config["data_loader_config"]
        self.accuracy = 0
        self.weighted_accuracy = 0
        self.binary_precision = []
        self.binary_recall = []
        self.preference_array = np.array([])
        self.observations = np.array([])
        self.top_10_barely_wrong = pd.DataFrame()
        self.top_10_barely_right = pd.DataFrame()
        self.top_10_totally_wrong = pd.DataFrame()
        self.combined_metrics = pd.DataFrame()

    def reset(self):
        """
        Resets the evaluation metrics and results.
        """
        self.actions = []
        self.probs = []
        self.labels = []
        self.labels_array = []
        self.accuracy = 0
        self.weighted_accuracy = 0
        self.binary_precision = []
        self.binary_recall = []
        self.preference_array = np.array([])
        self.observations = np.array([])
        self.top_10_barely_wrong = pd.DataFrame()
        self.top_10_barely_right = pd.DataFrame()
        self.top_10_totally_wrong = pd.DataFrame()
        self.combined_metrics = pd.DataFrame()
        self.eval_prefs = 0
        self.total_rewards = 0
        self.pref_loss = 0
        self.utility = 0
        self.hypervolume = 0
        self.reward_opt = None

    def _init_ufunc(self):
        """
        Initializes the utility function.

        Returns:
            function: A lambda function to compute utility.
        """
        return lambda x, y: tensor_to_numpy(
            self.agent.model.u_func(
                torch.tensor(x, device=self.agent.device, dtype=torch.float32),
                torch.tensor(y, device=self.agent.device, dtype=torch.float32),
            )
        )

    def comp_models(self):
        """
        Compares the current model with the best models based on hypervolume, utility, and preference loss.
        Updates the best models if the current model performs better.
        """
        for key, value in zip(
                ["best_hv", "best_utility", "best_pref_loss"],
                [self.hypervolume, self.utility, -self.pref_loss],
        ):
            if self.models[key]["value"] < value:
                self.models[key]["model"] = deepcopy(self.agent.model.state_dict())
                self.models[key]["value"] = deepcopy(value)

    def run_for_pref(self, pref, max_steps=np.inf, **kwargs):
        """
        Runs the evaluation for a given preference.

        Args:
            pref (list): The preference vector.
            max_steps (int, optional): Maximum number of steps to run the evaluation. Defaults to np.inf.

        Returns:
            np.array: Total reward accumulated.
        """
        self.env_eval.reset()
        total_reward = np.zeros(max(self.n_obj, 1))
        terminal = False
        # if pref is not None:
        #     print("Run: with Preference:", pref)

        while not terminal:
            observation = self.env_eval.observe()
            action, prob = self.agent.get_actions(observation.next_state, pref)
            terminal = observation.terminal or self.env_eval.steps >= max_steps
            # todo add loss calculation for classification since it is the accuracy but also might
            #  be interesting for other applications
            label = observation.label
            if isinstance(action, torch.Tensor):
                action = tensor_to_numpy(action)
            if isinstance(label, torch.Tensor):
                label = tensor_to_numpy(label)
            if isinstance(prob, torch.Tensor):
                prob = tensor_to_numpy(prob)

            total_reward += [-np.abs(prob - label).sum().item()] * max(self.n_obj, 1)

            if action.shape[0] == observation.next_state.shape[0]:
                if pref is None:
                    _pref = np.tile([], [action.shape[0], 1])
                else:
                    _pref = np.tile(pref, [action.shape[0], 1])
                self.preference_array = self._cat(self.preference_array, _pref)
                if self.is_img_data:
                    observation_ = observation.next_state.permute(0, 2, 3, 1)
                else:
                    observation_ = observation.next_state
                self.observations = self._cat(self.observations, observation_)
                self.actions = self._cat(self.actions, action)
                self.probs = self._cat(self.probs, prob)
                self.labels = self._cat(self.labels, label)

            if not terminal:
                self.env_eval.step()

        return total_reward

    def run(self, n_prefs, max_steps=np.inf):
        """
        Runs the evaluation for multiple preferences.

        Args:
            n_prefs (int): Number of preferences to evaluate.
            max_steps (int, optional): Maximum number of steps to run the evaluation. Defaults to np.inf.
        """
        self.reset()
        self.agent.eval_mode()
        if self.n_obj == 0:
            eval_prefs = [None]
        else:
            eval_prefs = get_eval_w(n_prefs, self.n_obj, self.agent.range_param)
        self.df_results = pd.DataFrame(columns=self.pref_cols + self.reward_cols, index=np.arange(len(eval_prefs)))

        with torch.no_grad():
            for n, eval_pref in enumerate(eval_prefs):
                # print("Run: ", n, "/", n_prefs)
                total_reward = self.run_for_pref(pref=eval_pref, max_steps=max_steps)
                # todo move the df_results to calc_metrics or another function
                if self.n_obj == 0:
                    self.df_results.loc[n, "pref"] = eval_pref
                    self.df_results.loc[n, "reward"] = total_reward[0]
                else:
                    self.df_results.loc[n, "utility"] = self.u_func(total_reward, eval_pref)
                    for obj in range(self.n_obj):
                        self.df_results.loc[n, f"pref_{obj}"] = eval_pref[obj]
                        self.df_results.loc[n, f"reward_{obj}"] = total_reward[obj]
        self.labels_array = self.labels.argmax(axis=1)
        self.calc_metrics()
        self.comp_models()

    @staticmethod
    def _cat(arr_all, arr):
        """
        Concatenates two arrays.

        Args:
            arr_all (np.array): The array to concatenate to.
            arr (np.array): The array to concatenate.

        Returns:
            np.array: The concatenated array.
        """
        if isinstance(arr, torch.Tensor):
            arr = tensor_to_numpy(arr.detach())
        if len(arr_all) == 0:
            arr_all = arr
        else:
            arr_all = np.concatenate([arr_all, arr], axis=0)
        return arr_all

    def calc_precision_recall(self, prediction, label):
        """
        Calculates precision and recall.

        Args:
            prediction (np.array): The predicted values.
            label (np.array): The true labels.

        Returns:
            tuple: Precision and recall values.
        """
        true_positives = (prediction * label).sum(axis=0)
        false_positives = (prediction * (1 - label)).sum(axis=0)
        false_negatives = ((1 - prediction) * label).sum(axis=0)
        precision = true_positives / (true_positives + false_positives + self.EPSILON)
        recall = true_positives / (true_positives + false_negatives + self.EPSILON)
        return precision, recall

    def calc_metrics(self):
        """
        Calculates various evaluation metrics.
        """
        if self.is_img_data:
            log_metrics = DataFrameAnalyzer(self.labels, self.probs)
            self.top_10_barely_wrong = log_metrics.dfTOP10_bw
            self.top_10_barely_right = log_metrics.dfTOP10_br
            self.top_10_totally_wrong = log_metrics.dfTOP10_tw
            self.combined_metrics = log_metrics.df_combined_metrics

        self.eval_prefs = self.df_results[self.pref_cols].values
        self.total_rewards = self.df_results[self.reward_cols].values
        # todo calc precision and recall normally and weighted

        self.accuracy = (self.labels_array == self.actions).mean()
        precision, recall = self.calc_precision_recall(self.probs, self.labels)

        if self.agent.is_moo:

            weighted_labels = self.labels * self.preference_array
            weighted_labels /= weighted_labels.sum(axis=1, keepdims=True).clip(min=self.EPSILON)

            self.weighted_accuracy = (weighted_labels.argmax(axis=1) == self.actions).mean()
            weighted_precision, weighted_recall = self.calc_precision_recall(self.probs, weighted_labels)
        else:
            self.weighted_accuracy = self.accuracy
            weighted_precision, weighted_recall = precision, recall

        self.binary_precision = precision
        self.binary_recall = recall
