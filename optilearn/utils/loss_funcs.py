from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor, nn

from optilearn.evaluation import metrics


class AbstractLoss(ABC):
    def __init__(self, **kwargs):
        self._criterion: Callable = lambda x: x

    @abstractmethod
    def __call__(self, actions, targets, preferences=None, weights=None, **kwargs):
        pass

    @property
    def criterion(self):
        return self._criterion


class WeightedCrossEntropyLoss(AbstractLoss):
    def __init__(self):
        super().__init__()
        self._criterion = nn.CrossEntropyLoss()

    def __call__(self, actions, targets, preferences=None, critical_class_mask=None, **kwargs):

        if any(critical_class_mask):
            weighted_targets = metrics.apply_criticality_preference(
                targets,
                preferences,
                critical_class_mask=critical_class_mask,
            )
        else:
            weighted_targets = targets

        # return self.criterion(preds, weighted_targets)
        return self.criterion(actions, weighted_targets.argmax(dim=1))


class WeightedClassLoss(AbstractLoss):
    def __init__(self):
        super().__init__()

    def __call__(self, actions, targets, preferences=None, critical_class_mask=None, **kwargs):
        critical_class_mask_tensor = torch.tensor(
            critical_class_mask, device=preferences.device, dtype=torch.int8
        ).unsqueeze(1)
        entropy = -targets * torch.log(actions.clip(min=1e-6))
        crit_weights = preferences[:, 0] * critical_class_mask_tensor
        non_crit_weights = preferences[:, 1] * (1 - critical_class_mask_tensor)
        weights = (crit_weights + non_crit_weights).T
        return (weights * entropy).sum(dim=1).mean()


class FBetaLoss(AbstractLoss):
    def __init__(self):
        super().__init__()

    def __call__(self, actions, targets, preferences=None, critical_class_mask=None, **kwargs):
        # Apply weights based on preferences and critical class mask
        critical_class_mask_tensor = torch.tensor(
            critical_class_mask, device=preferences.device, dtype=torch.int8
        ).unsqueeze(1)
        # Compute F1 score for each class
        epsilon = 1e-7  # Smoothing term to avoid division by zero

        pref0 = preferences[:, 0]
        pref1 = preferences[:, 1]
        n_prefs = len(pref0.unique())

        # calculate the true positives, false positives and false negatives
        prediction = torch.round(actions[:, 0])
        label = torch.round(targets[:, 0])

        true_positives = (prediction * label).reshape(n_prefs, -1).sum(dim=1)
        false_positives = (prediction * (1 - label)).reshape(n_prefs, -1).sum(dim=1)
        false_negatives = ((1 - prediction) * label).reshape(n_prefs, -1).sum(dim=1)
        precision = true_positives / (true_positives + false_positives + epsilon)
        recall = true_positives / (true_positives + false_negatives + epsilon)

        beta = (pref0 / pref1).reshape(n_prefs, -1).mean(dim=1)
        f_beta_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + epsilon)
        return 1 - f_beta_score.mean()


class CustomCrossEntropyLoss(AbstractLoss):
    def __init__(self):
        super().__init__()
        self._criterion = nn.CrossEntropyLoss()

    def __call__(self, actions, targets, preferences=None, **kwargs):
        return self.criterion(actions, targets.argmax(dim=1))


class BinaryPrecisionRecallLoss(AbstractLoss):
    def __init__(self, weight_rewards: bool = True, **kwargs):
        self.weight_rewards = weight_rewards
        super().__init__(**kwargs)

    def __call__(
        self,
        actions: Tensor,
        targets: Tensor,
        preferences: Tensor = None,
        rewards=None,
        **kwargs,
    ):
        """
        Objective -> J: w.R.a'
        preference[0] --> precision and preference[1] --> recall

        w : 1 x N_obj
        R : N_obj x N_classes
        a': N_classes x 1

        Parameters
        ----------
        actions
        preferences
        rewards:  reward from the environment

        Returns
        -------

        """

        # labels = labels + preds

        pr_reward = metrics.get_binary_precision_recall_matrix(targets)

        if self.weight_rewards:
            pr_reward = pr_reward * targets.unsqueeze(1)
        # reward = reward[:, :, :1]

        # Reshaping to Batch x (1 x N_obj)
        preferences = preferences.unsqueeze(dim=1)

        # Reshaping to Batch x (N_classes x 1)
        actions = actions.unsqueeze(dim=1).transpose(1, 2)
        # preds = preds[:, :1, :]

        objective = torch.bmm(
            torch.bmm(preferences, pr_reward.detach()),
            actions,
        ).squeeze()

        return -objective


class NaiveCostLoss(AbstractLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, actions, targets, preferences=None, critical_indices=None, **kwargs):
        costs = self._get_costs(targets, preferences)
        norm_costs = costs / costs.sum(dim=-1).unsqueeze(-1)

        return ((norm_costs - actions) ** 2).sum(dim=-1)

    @staticmethod
    def _get_costs(labels, preferences):
        return (labels) * preferences
