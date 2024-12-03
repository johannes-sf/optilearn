import torch
from torch import nn as nn
from torch.nn import functional as F


def get_func(fn):
    """
    Get the activation function based on the given name.

    Args:
        fn (str): The name of the activation function.

    Returns:
        function: The activation function.

    Raises:
        Exception: If the activation function name is invalid.

    """
    if fn == "tanh":
        return torch.tanh
    elif fn == "sigmoid":
        return torch.sigmoid
    elif fn == "relu":
        return torch.relu
    elif fn == "lrelu":
        return torch.nn.LeakyReLU()
    elif fn == "softplus":
        return F.softplus
    elif fn == "linear":
        return lambda x: x
    else:
        raise Exception("wrong activation function")


@torch.no_grad()
def set_init(moduls, method="orthogonal", weight_gain=1):
    """
    Set the initial weights of the modules.

    Args:
        moduls: The modules.
        method (str, optional): The weight initialization method. Defaults to "orthogonal".
        weight_gain (int, optional): The weight gain. Defaults to 1.

    """
    for modul in moduls:
        if modul is None:
            continue
        for layer in modul:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                if method == "xavier":
                    nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
                elif method == "orthogonal":
                    nn.init.orthogonal_(layer.weight, gain=weight_gain)
                elif method == "kaiming":
                    nn.init.kaiming_normal_(layer.weight)
                else:
                    nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                nn.init.constant_(layer.bias, 0.0)


@torch.no_grad()
def init_layers(modules):
    """
    Initialize the layers of the modules.

    Args:
        modules: The modules.

    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
