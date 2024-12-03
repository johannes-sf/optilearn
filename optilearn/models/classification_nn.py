from typing import Type

import numpy as np
import torch
from torch import Tensor

from optilearn.models.abstract_model import AbstractModel
from optilearn.models.nn_generators.custom_nn import CustomNN
from optilearn.utils.loss_funcs import AbstractLoss
from optilearn.utils.u_funcs import AbstractUFunc


class ClassificationNN(AbstractModel):
    def __init__(
        self,
        config: dict,
        u_func: Type[AbstractUFunc] = None,
        loss_criterion: Type[AbstractLoss] = None,
        device: str = "cpu",
    ):
        super(ClassificationNN, self).__init__(
            config,
            u_func=u_func,
            loss_criterion=loss_criterion,
            device=device,
        )

        self.nn = CustomNN(
            dim_in=config["s_dim"],
            dim_out=config["a_dim"],
            dim_p=config["p_dim"],
            nn_config=config["model_config"],
            device=device,
        )

        # ToDo: Add optimizer from config?
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=config["model_config"]["lr"])

    def loss_func(
        self,
        states: Tensor,
        preds: Tensor,
        labels: Tensor,
        preferences: Tensor,
    ) -> Tensor:
        """
        Calculates the loss by the criterion defined by <self.loss_criterion> i.e.
        cross-entropy loss.

        As the reward tensor returned by image classification env is the difference
        between the one hot class labels and the predicted class probabilities i.e.
        pred tensor, the one hot labels are reconstructed by adding the
        reward tensor to the preds tensor.


        Parameters
        ----------
        states
        preferences
        preds
        labels

        Returns
        -------

        """
        # todo non moo behaviour
        # ToDo: Account for critical/non-critical class mapping
        if self.loss_type == "weighted_cross_entropy":
            pass

        loss = self.loss_criterion(preds, labels, preferences=preferences, critical_class_mask=self.critical_class_mask)

        return loss

    def forward(self, state, pref):
        x = self.nn.forward(state, pref)
        x = torch.softmax(x, dim=1)
        return x

    def choose_action(self, state, pref, **kwargs):
        with torch.no_grad():
            probs = self.forward(state, pref)
            return probs.argmax(dim=1), probs

    def make_step(self, states, labels, preds, prefs):
        self.optim.zero_grad()
        loss = self.loss_func(states, preds, labels, prefs)
        loss = loss.mean()
        loss.backward()
        self.optim.step()
        return loss.cpu().detach()
