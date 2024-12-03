from typing import Type

import numpy as np
import torch

from optilearn.environments.abstract_env import AbstractEnv
from optilearn.models import AbstractModel
from optilearn.utils.loss_funcs import AbstractLoss
from optilearn.utils.u_funcs import AbstractUFunc
from optilearn.utils.utils import tensor_to_numpy, tensor


class Agent:
    def __init__(
            self,
            config: dict,
            model: Type[AbstractModel],
            u_func: Type[AbstractUFunc] = None,
            loss_criterion: Type[AbstractLoss] = None,
            device="cpu",
    ):
        """
        Initializes the Agent.

        Args:
            config (dict): Configuration dictionary.
            model (Type[AbstractModel]): Model class.
            u_func (Type[AbstractUFunc], optional): Utility function class. Defaults to None.
            loss_criterion (Type[AbstractLoss], optional): Loss criterion class. Defaults to None.
            device (str, optional): Device specification. Defaults to "cpu".
        """
        self.is_moo = config["p_dim"] > 1
        self.training_steps = 0
        self.n_obj = config["p_dim"]
        if not self.is_moo:
            config["p_dim"] = 0
        self.range_param = config["pref_range"] if "pref_range" in config.keys() else 1
        self.sample_alpha = config["sample_alpha"] if "sample_alpha" in config else 1
        self.device = device
        self.model = model(config, u_func=u_func, loss_criterion=loss_criterion, device=device)
        self.torch_interface = config["model_config"].get("torch_interface", False)
        self.performance_metric = None

    def eval_mode(self):
        """
        Sets the model to evaluation mode.
        """
        self.model.eval()

    def train_mode(self):
        """
        Sets the model to training mode.
        """
        self.model.train()

    def sample_pref(self, n_prefs, alpha=4):
        """
        Generates preference vectors using a Dirichlet distribution.

        Args:
            n_prefs (int): Number of preferences to sample.
            alpha (int, optional): Alpha parameter for the Dirichlet distribution. Defaults to 4.

        Returns:
            torch.Tensor: Sampled preference vectors.
        """
        prefs = torch.tensor(
            np.random.dirichlet([alpha] * self.n_obj, n_prefs), dtype=torch.float32, device=self.device
        )

        return (1 - self.range_param) / 2 + self.range_param * prefs

    # ToDo: pass DataLoaderResponse, instead of the entire env
    def train(self, env: AbstractEnv, n_prefs: int = None, **kwargs):
        """
        Trains the model using the provided environment.

        Args:
            env (AbstractEnv): The environment to train on.
            n_prefs (int, optional): Number of preferences to sample. Defaults to None.

        Returns:
            tuple: Loss, accuracy, and None.
        """
        observation = env.observe()
        if observation.terminal:
            env.reset()
        if self.torch_interface:
            self.train_mode()
            accuracy, loss = self._train_with_torch_interface(observation, n_prefs)
        else:
            accuracy, loss = self._train_sk_interface(observation, n_prefs)
        self.training_steps += 1
        if not env.terminal:
            env.step()
        return loss, accuracy, None

    def _extend(self, prefs, states, labels, n_prefs):
        """
        Repeats and reshapes preference vectors, states, and labels.

        Args:
            prefs (torch.Tensor): Preference vectors.
            states (torch.Tensor): States.
            labels (torch.Tensor): Labels.
            n_prefs (int): Number of preferences.

        Returns:
            tuple: Extended preferences, states, and labels.
        """
        #todo fix for numpy
        state_shape = states.shape
        prefs = prefs.repeat([1, state_shape[0]]).reshape([-1, prefs.shape[-1]])
        states = states.repeat([n_prefs] + [1] * (len(state_shape) - 1)).reshape([-1] + list(state_shape[1:]))
        labels = labels.repeat([n_prefs, 1]).reshape([-1, labels.shape[-1]])
        return prefs, states, labels

    def _train_with_torch_interface(self, observation, n_prefs):
        """
        Trains the model using a PyTorch interface.

        Args:
            observation (Observation): The observation from the environment.
            n_prefs (int): Number of preferences to sample.

        Returns:
            tuple: Accuracy and loss.
        """
        state = observation.next_state
        label = observation.label

        if self.is_moo:
            prefs = self.sample_pref(n_prefs, self.sample_alpha)
            prefs, states, labels = self._extend(prefs, state, label, n_prefs)
            preds = self.model.forward(states, pref=prefs)
        else:
            labels = label
            prefs = None
            states = state
            preds = self.model.forward(state, pref=None)

        loss = self.model.make_step(states=states, preds=preds, labels=labels, prefs=prefs)
        accuracy = tensor_to_numpy((labels.argmax(dim=-1) == preds.argmax(dim=-1))).mean()

        return accuracy, loss.item()

    def _train_sk_interface(self, observation, n_prefs):
        """
        Trains the model using a scikit-learn interface.

        Args:
            observation (Observation): The observation from the environment.
            n_prefs (int): Number of preferences to sample.

        Returns:
            tuple: Accuracy and loss.
        """
        state = observation.next_state
        labels = observation.label

        # ToDo: Implement MOO scenario
        if self.is_moo:
            pass

        prefs = None

        loss = self.model.make_step(states=state, labels=labels, prefs=prefs, preds=None)
        preds, probs = self.model.choose_action(state=state, pref=prefs)
        accuracy = float((labels.argmax(axis=1) == preds).mean())
        return accuracy, loss

    def get_actions(self, state, pref, **kwargs):
        """
        Retrieves actions from the model based on the current state and preferences.

        Args:
            state (torch.Tensor): The current state.
            pref (torch.Tensor): The preference vector.

        Returns:
            Any: The chosen action.
        """
        # todo fix or make more generic
        if self.torch_interface:
            state = tensor(state, device=self.device, dtype=torch.float32)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            if pref is not None:
                pref = tensor(pref, device=self.device, dtype=torch.float32)
                if len(state.shape) == 1:
                    pref = pref.unsqueeze(0)
                pref = pref.repeat([state.shape[0], 1])
        return self.model.choose_action(state, pref)
