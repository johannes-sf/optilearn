from abc import abstractmethod
from typing import Type

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from optilearn.utils.loss_funcs import AbstractLoss
from optilearn.utils.u_funcs import AbstractUFunc


class AbstractModel(nn.Module):
    """
Abstract base class for models in the moo_classification project.

Attributes
----------
s_dim : int
    Dimension of the state space.
a_dim : int
    Dimension of the action space.
p_dim : int
    Dimension of the preference space.
loss_criterion : AbstractLoss
    Loss function used for training the model.
device : str
    Device to run the model on (e.g., 'cpu' or 'cuda').
u_func : AbstractUFunc
    Utility function used in the model.
"""

    def __init__(
            self,
            config: dict,
            u_func: Type[AbstractUFunc] = None,
            loss_criterion: Type[AbstractLoss] = None,
            device: str = "cpu",
    ):
        """
            Initialize the AbstractModel.

            Parameters
            ----------
            config : dict
                Configuration dictionary containing model parameters.
            u_func : Type[AbstractUFunc], optional
                Utility function class to be used in the model.
            loss_criterion : Type[AbstractLoss], optional
                Loss function class to be used in the model.
            device : str, optional
                Device to run the model on (default is 'cpu').
            """
        super(AbstractModel, self).__init__()
        self.s_dim = config["s_dim"]
        self.a_dim = config["a_dim"]
        self.p_dim = config["p_dim"]
        self.loss_criterion = loss_criterion()
        self.u_func = u_func()
        self.label_to_class_index = config["label_to_class_index"]
        self.class_index_to_label = config["class_index_to_label"]
        self.critical_classes = config["critical_classes"]
        self.critical_class_mask = np.array(config["critical_class_mask"]).astype(bool)
        self.loss_type = config["loss_type"]

        self.device = device

        # @abstractmethod

    def forward(self, x: Tensor, pref) -> Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : Tensor
            Input tensor.
        pref : Any
            Preferences for the model.

        Returns
        -------
        Tensor
            Output tensor.
        """
        return torch.tensor([])

    @abstractmethod
    def choose_action(self, state, pref):
        """
         Choose an action based on the current state and preferences.

         Parameters
         ----------
         state : Any
             Current state.
         pref : Any
             Preferences for the model.
         epsilon : float, optional
             Exploration rate (default is None).
         past_reward : Any, optional
             Past reward information (default is None).

         Returns
         -------
         action : Any
             Chosen action.
         probs : Any
             Probabilities associated with the action.
         """
        action = None
        probs = None
        return action, probs

    # @abstractmethod
    def loss_func(self, *args, **kwargs):
        """
        Compute the loss for the model.

        Parameters
        ----------
        *args : tuple
            Positional arguments.
        **kwargs : dict
            Keyword arguments.
        """
        pass

    @abstractmethod
    def make_step(self, states, labels, preds, prefs):
        """
        Perform a single step of optimization.

        Parameters
        ----------
        *args : tuple
            Positional arguments.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        loss : Any
            Total loss.
        loss_1 : Any
            First component of the loss.
        loss_2 : Any
            Second component of the loss.
        """
        loss = None
        loss_1 = None
        loss_2 = None
        return loss, loss_1, loss_2
