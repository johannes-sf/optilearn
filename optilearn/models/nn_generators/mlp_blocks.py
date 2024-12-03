import torch.nn as nn
from optilearn.models.utils.generator_utils import get_func


class BasicMlpBlock(nn.Module):
    """

    """
    def __init__(self, d_in, size, act):
        super().__init__()
        self.linear = nn.Linear(d_in, size)
        self.act = get_func(act)
        self.out_dim = size

    def forward(self, x):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        x = self.linear(x)
        x = self.act(x)
        return x
