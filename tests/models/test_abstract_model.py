import torch
from optilearn.models.abstract_model import AbstractModel
from tests.models.model_fixtures import mock_u_func, mock_loss_func, BASE_CONFIG


def test_abstract_model_forward(mock_u_func, mock_loss_func):
    model = AbstractModel(config=BASE_CONFIG, u_func=mock_u_func, loss_criterion=mock_loss_func)
    x = torch.zeros(1, BASE_CONFIG["s_dim"])
    pref = torch.zeros(1, BASE_CONFIG["p_dim"])
    output = model.forward(x, pref)
    assert output.shape == torch.Size([0])


def test_abstract_model_choose_action(mock_u_func, mock_loss_func):
    model = AbstractModel(BASE_CONFIG, u_func=mock_u_func, loss_criterion=mock_loss_func)
    state = torch.zeros(1, BASE_CONFIG["s_dim"])
    pref = torch.zeros(1, BASE_CONFIG["p_dim"])
    action, probs = model.choose_action(state, pref)
    assert action is None  # Replace None with the expected action
    assert probs is None  # Replace None with the expected probabilities