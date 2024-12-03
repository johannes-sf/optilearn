import lightgbm as lgb
import numpy as np
from typing import Type
import torch
from lightgbm import LGBMClassifier
from optilearn.models.abstract_model import AbstractModel
from optilearn.utils.loss_funcs import AbstractLoss
from optilearn.utils.u_funcs import AbstractUFunc
from sklearn.model_selection import train_test_split
from torch import Tensor
from treeboost_autograd import LightGbmObjective


# This is meant to be used by treeboost_autograd
def absolute_error_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculates the absolute error loss.

    Args:
        preds (torch.Tensor): The predicted values.
        targets (torch.Tensor): The target values.

    Returns:
        torch.Tensor: The calculated loss.
    """
    return torch.mean((preds - targets) ** 2)


def absolute_error_loss_eval(preds, targets):
    """
    Evaluation function for absolute error loss.

    Args:
        preds: The predicted values.
        targets: The target values.

    Returns:
        tuple: A tuple containing the metric name, value, and a boolean indicating if higher is better.
    """
    metric_name = "abs_error"
    value = np.mean((preds - targets) ** 2)
    is_higher_better = False

    return metric_name, value, is_higher_better


class ClassificationLGBM(AbstractModel):
    """
    A classification model using LightGBM, inheriting from AbstractModel.

    Attributes:
        objective (LightGbmObjective): The custom LightGBM objective function.
        model (LGBMClassifier): The LightGBM model used for classification.
        label_to_class_index (dict): Mapping from labels to class indices.
        class_index_to_label (dict): Mapping from class indices to labels.
        critical_classes (list): List of critical classes.
        critical_class_mask (np.ndarray): Mask for critical classes.
        loss_type (str): Type of loss function to use.
        device (str): Device to run the model on.
    """

    def __init__(self, config: dict,
                 u_func: Type[AbstractUFunc] = None,
                 loss_criterion: Type[AbstractLoss] = None,
                 device: str = "cpu", **kwargs):
        """
        Initializes the ClassificationLGBM model.

        Args:
            config (dict): Configuration dictionary.
            device (str): Device to run the model on.
            **kwargs: Additional keyword arguments.
        """
        super(ClassificationLGBM, self).__init__(
            config,
            u_func=u_func,
            loss_criterion=loss_criterion,
            device=device,
        )
        self.objective = LightGbmObjective(loss_function=absolute_error_loss)
        # todo move params to params in config
        params = config["model_config"]["params"]
        self.is_moo = config["p_dim"] > 1
        if self.is_moo:
            params["objective"] = self.objective
        self.model: LGBMClassifier = LGBMClassifier(**params)


    def loss_func(
            self,
            states: Tensor,
            actions: Tensor,
            rewards: Tensor,
            preferences: Tensor,
    ) -> Tensor:
        """
        Calculates the loss by the criterion defined by <self.loss_criterion> i.e.
        cross-entropy loss.

        As the reward tensor returned by image classification env is the difference
        between the one hot class labels and the predicted class probabilities i.e.
        pred tensor, the one hot labels are reconstructed by adding the
        reward tensor to the preds tensor.

        Args:
            states (Tensor): The input states.
            actions (Tensor): The predicted class probabilities.
            rewards (Tensor): The true class labels.
            preferences (Tensor): The preference tensor.

        Returns:
            Tensor: The calculated loss.
        """

        # ToDo: Account for critical/non-critical class mapping
        if self.loss_type == "weighted_cross_entropy":
            pass

        labels = (rewards + actions).detach()
        loss = self.loss_criterion(
            actions, labels, preferences=preferences, critical_class_mask=self.critical_class_mask
        )

        return loss

    def forward(self, state, pref):
        """
        Forward pass to get the predicted class probabilities.

        Args:
            state: The input state.
            pref: The input preferences.

        Returns:
            np.ndarray: The predicted class probabilities.
        """
        if self.is_moo:
            # todo implement moo stuff here
            pass
            return self.model.predict_proba(state)
        else:
            return self.model.predict_proba(state)

    def choose_action(self, state, pref):
        """
        Chooses an action based on the input state.

        Args:
            state: The input state.
            pref: The input preferences.

        Returns:
            np.ndarray: The predicted class labels.
        """
        probs = self.forward(state, pref)
        return probs.argmax(axis=1), probs

    def make_step(self, states, labels, preds=None, prefs=None):
        """
        Makes a step in the model training or evaluation.

        Args:
            states: The input states.
            labels: The true class labels.
            preds: The predicted values (optional).
            prefs: The preference tensor (optional).

        Returns:
            float: The negative score as the loss.
        """
        # ToDo: Add eval metric
        # ToDo: Partially initialize existing losses with prefs
        # batch_size = 100000
        #
        # for start_index in tqdm(range(0, states.shape[0], batch_size)):
        #     # Get the current batch
        #     end = min(start_index + batch_size, states.shape[0])
        #     state_batch, label_batch = states[start_index:end], labels[start_index:end]
        #
        #     if self.model.__sklearn_is_fitted__():
        #         self.model = self.model.fit(
        #             state_batch, label_batch.tolist(), eval_metric="logloss", init_model=self.model.booster_
        #         )
        #
        #     else:
        #         self.model = self.model.fit(state_batch, label_batch.tolist(), eval_metric="logloss")

        # x_train, x_test, y_train, y_test = train_test_split(states, labels, test_size=0.1)

        self.model.fit(
            X=states,
            y=labels.argmax(axis=1),
            eval_metric=absolute_error_loss_eval,
            callbacks=[lgb.callback.log_evaluation()],
        )

        loss = -self.model.score(
            X=states,
            y=labels.argmax(axis=1)
        )
        return loss
