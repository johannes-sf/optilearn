import pytest
from unittest.mock import Mock
import numpy as np
from optilearn.utils.u_funcs import AbstractUFunc, Linear
from optilearn.utils.loss_funcs import AbstractLoss, CustomCrossEntropyLoss

BATCH_SIZE = 4
BASE_CONFIG = {
    "s_dim": 10,
    "a_dim": 2,
    "p_dim": 3,
    "label_to_class_index": {0: 0, 1: 1},
    "class_index_to_label": {0: 0, 1: 1},
    "critical_classes": [0],
    "critical_class_mask": [1, 0],
    "loss_type": "cross_entropy",
    "model_config": {}
}


def get_mock_values(config):
    aux_states = np.mod(np.arange(BATCH_SIZE), config["s_dim"])
    states = np.zeros([BATCH_SIZE, config["s_dim"]])
    states[np.arange(aux_states.size), aux_states] = 1
    preds = np.zeros([BATCH_SIZE, config["a_dim"]])
    one_hot_labels = np.mod(np.arange(BATCH_SIZE), config["a_dim"])
    labels = np.zeros([BATCH_SIZE, config["a_dim"]])
    labels[np.arange(one_hot_labels.size), one_hot_labels] = 1
    prefs = np.zeros([BATCH_SIZE, config["p_dim"]])
    return states, preds, labels, prefs


@pytest.fixture
def mock_u_func():
    # Linear.__call__ = Mock(return_value=torch.tensor([0.5, 0.5]))
    return Linear


@pytest.fixture
def mock_loss_func():
    # CustomCrossEntropyLoss.__call__ = Mock(return_value=torch.tensor([0.5, 0.5]))
    return CustomCrossEntropyLoss
