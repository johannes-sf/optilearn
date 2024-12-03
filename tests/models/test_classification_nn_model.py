import pytest
import torch
from optilearn.models.classification_nn import ClassificationNN
from tests.models.model_fixtures import mock_u_func, mock_loss_func, BASE_CONFIG, BATCH_SIZE


@pytest.fixture
def mock_resnet_config():
    BASE_CONFIG["s_dim"] = 3  # channel dimensions
    BASE_CONFIG["model_config"] = {
        "lr": 0.001,
        "model_type": "resnet",
        "torch_interface": True,
        "input_size": [28, 28],
        "feature_extractor": {
            "pretrained_block": True,
            "pretrained_model_name": "resnet18",
            "fine_tune": True,
            "cnn": {}
        },
        "mlp": {
            "attention_head": False,
            "mlp": {
                0: {
                    "act": "lrelu",
                    "size": 32
                }
            }
        }
    }
    return BASE_CONFIG


@pytest.fixture
def mock_cnn_config():
    BASE_CONFIG["s_dim"] = 3  # channel dimensions
    BASE_CONFIG["model_config"] = {
        "lr": 0.001,
        "model_type": "nn",
        "torch_interface": True,
        "input_size": [28, 28],
        "feature_extractor": {
            "pretrained_block": False,
            "pretrained_model_name": None,
            "fine_tune": False,
            "cnn": {
                0: {
                    "act": "lrelu",
                    "batch_norm": True,
                    "bias": False,
                    "block": "basic_cnn",
                    "kernel": 3,
                    "padding": 1,
                    "pooling": 1,
                    "size": 3
                }
            },
        },
        "mlp": {
            "attention_head": False,
            "mlp": {
                0: {
                    "act": "lrelu",
                    "size": 32
                }
            }
        }
    }
    return BASE_CONFIG


@pytest.fixture
def mock_mlp_config():
    BASE_CONFIG["model_config"] = {
        "lr": 0.001,
        "model_type": "mlp",
        "torch_interface": True,
        "mlp": {
            "attention_head": False,
            "mlp": {
                0: {
                    "act": "lrelu",
                    "size": 32
                }
            }
        }
    }
    return BASE_CONFIG


def test_classification_nn_forward(mock_mlp_config, mock_u_func, mock_loss_func):
    model = ClassificationNN(config=mock_mlp_config, u_func=mock_u_func, loss_criterion=mock_loss_func)
    x = torch.zeros(BATCH_SIZE, mock_mlp_config["s_dim"])
    pref = torch.zeros(BATCH_SIZE, mock_mlp_config["p_dim"])
    output = model.forward(x, pref)
    assert output.shape == torch.Size([BATCH_SIZE, mock_mlp_config["a_dim"]])
    assert ((output >= 0) & (output <= 1)).all()
    assert (output.sum(dim=1) > 0.9999).all()


def test_classification_nn_choose_action(mock_mlp_config, mock_u_func, mock_loss_func):
    model = ClassificationNN(config=mock_mlp_config, u_func=mock_u_func, loss_criterion=mock_loss_func)
    state = torch.zeros(BATCH_SIZE, mock_mlp_config["s_dim"])
    pref = torch.zeros(BATCH_SIZE, mock_mlp_config["p_dim"])
    action, probs = model.choose_action(state, pref)
    assert action.shape == torch.Size([BATCH_SIZE])
    assert action.dtype == torch.int64
    assert ((action >= 0) & (action <= mock_mlp_config["a_dim"])).all()
    assert probs.shape == torch.Size([BATCH_SIZE, mock_mlp_config["a_dim"]])
    assert ((probs >= 0) & (probs <= 1)).all()
    assert (probs.sum(dim=1) > 0.9999).all()


def test_classification_nn_loss_func(mock_mlp_config, mock_u_func, mock_loss_func):
    model = ClassificationNN(config=mock_mlp_config, u_func=mock_u_func, loss_criterion=mock_loss_func)
    states = torch.zeros(BATCH_SIZE, mock_mlp_config["s_dim"])
    preds = torch.zeros(BATCH_SIZE, mock_mlp_config["a_dim"])
    labels = torch.zeros(BATCH_SIZE, mock_mlp_config["a_dim"])
    prefs = torch.zeros(BATCH_SIZE, mock_mlp_config["p_dim"])
    loss = model.loss_func(states, preds, labels, prefs)
    assert loss.shape == torch.Size([])
    assert loss.dtype == torch.float32


def test_classification_nn_make_step(mock_mlp_config, mock_u_func, mock_loss_func):
    model = ClassificationNN(config=mock_mlp_config, u_func=mock_u_func, loss_criterion=mock_loss_func)
    states = torch.zeros(BATCH_SIZE, mock_mlp_config["s_dim"], requires_grad=True)
    preds = torch.zeros(BATCH_SIZE, mock_mlp_config["a_dim"], requires_grad=True)
    labels = torch.zeros(BATCH_SIZE, mock_mlp_config["a_dim"], requires_grad=True)
    prefs = torch.zeros(BATCH_SIZE, mock_mlp_config["p_dim"], requires_grad=True)
    loss = model.make_step(states, preds, labels, prefs)
    assert loss.shape == torch.Size([])
    assert loss.dtype == torch.float32


def test_classification_nn_img_choose_action(mock_cnn_config, mock_resnet_config, mock_u_func, mock_loss_func):
    prefs = torch.zeros(BATCH_SIZE, mock_cnn_config["p_dim"], requires_grad=True)
    for mock_config in [mock_cnn_config, mock_resnet_config]:
        states = torch.zeros(*[BATCH_SIZE, mock_config["s_dim"]] + mock_config["model_config"]["input_size"])
        model = ClassificationNN(config=mock_config, u_func=mock_u_func, loss_criterion=mock_loss_func)
        action, probs = model.choose_action(states, prefs)
        assert action.shape == torch.Size([BATCH_SIZE])
        assert action.dtype == torch.int64
        assert ((action >= 0) & (action <= mock_config["a_dim"])).all()
        assert probs.shape == torch.Size([BATCH_SIZE, mock_cnn_config["a_dim"]])
        assert ((probs >= 0) & (probs <= 1)).all()
        assert (probs.sum(dim=1) > 0.9999).all()

def test_classification_nn_img_make_step(mock_cnn_config, mock_resnet_config, mock_u_func, mock_loss_func):
    preds = torch.zeros(BATCH_SIZE, mock_cnn_config["a_dim"], requires_grad=True)
    labels = torch.zeros(BATCH_SIZE, mock_cnn_config["a_dim"], requires_grad=True)
    prefs = torch.zeros(BATCH_SIZE, mock_cnn_config["p_dim"], requires_grad=True)
    for mock_config in [mock_cnn_config, mock_resnet_config]:
        states = torch.zeros(*[BATCH_SIZE, mock_config["s_dim"]] + mock_config["model_config"]["input_size"],
                             requires_grad=True)
        model = ClassificationNN(config=mock_config, u_func=mock_u_func, loss_criterion=mock_loss_func)
        loss = model.make_step(states, preds, labels, prefs)
        assert loss.shape == torch.Size([])
        assert loss.dtype == torch.float32
