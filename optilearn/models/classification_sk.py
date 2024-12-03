import numpy as np
from torch import Tensor

from optilearn.models.abstract_model import AbstractModel
from sklearn.ensemble import RandomForestClassifier


class ClassificationSK(AbstractModel):
    """
    A response class for the TabularDataLoader containing the state and optional state labels.

    Attributes:
        state (DataFrame): The state data.
        state_labels (Optional[Series]): The state labels, default is None.
    """

    def __init__(self, config: dict, u_func, loss_criterion, device: str = "cpu", **kwargs):
        """
        Initializes the ClassificationSK model.

        Args:
            config (dict): Configuration dictionary.
            u_func: Utility function.
            loss_criterion: Loss criterion function.
            device (str): Device to run the model on.
        """
        super(ClassificationSK, self).__init__(config, u_func, loss_criterion, device)

        # todo fix loss
        # https://stackoverflow.com/questions/56682250/setting-a-custom-loss-for-sklearn-gradient-boosting-classfier
        # might be a bit more comlpicated
        self.is_moo = config["p_dim"] > 1
        if self.is_moo:
            #todo implement moo stuff here
            pass
        #todo incorporate these
        self.s_dim = config["s_dim"]
        self.a_dim = config["a_dim"]
        self.p_dim = config["p_dim"]
        self.model = self._init_model(config["model_config"])

    @staticmethod
    def _init_model(config):
        """
        Initializes the scikit-learn model based on the given configuration.

        Args:
            config (dict): Configuration dictionary for the model.

        Returns:
            RandomForestClassifier: The initialized scikit-learn model.
        """
        model_name = config["sk_model_name"]
        if model_name == "random_forest":
            model = RandomForestClassifier(**config["params"])
        else:
            raise ValueError(f"Unknown model_name given: {model_name}")
        return model

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

        Args:
            states (Tensor): The input states.
            preds (Tensor): The predicted class probabilities.
            labels (Tensor): The true class labels.
            preferences (Tensor): The preference tensor.

        Returns:
            Tensor: The calculated loss.
        """
        loss = self.loss_criterion(
            preds, labels, preferences=preferences, critical_class_mask=self.critical_class_mask
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
            #todo implement moo stuff here
            pass
            return self.model.predict_proba(state)
        else:
            return self.model.predict_proba(state)

    def choose_action(self, state, pref, **kwargs):
        """
        Chooses an action based on the input state and preferences.

        Args:
            state: The input state.
            pref: The input preferences.
            **kwargs: Additional arguments.

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
            **kwargs: Additional arguments.

        Returns:
            float: The negative score as the loss.
        """
        self.model = self.model.fit(states, labels.argmax(axis=1))
        loss = -self.model.score(states, labels.argmax(axis=1))
        return loss
