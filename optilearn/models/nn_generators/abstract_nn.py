import torch
from copy import deepcopy
from typing import Optional


class AbstractNN(torch.nn.Module):
    """

    """
    def __init__(self, dim_in, dim_p, dim_out, nn_config):
        super(AbstractNN, self).__init__()
        self.dim_in = deepcopy(dim_in)
        self.dim_p = deepcopy(dim_p)
        self.dim_out = dim_out
        self.nn_config = nn_config

    # @torch.compile(backend="aot_eager")
    def forward(self, x, pref: Optional[torch.Tensor]=None):
        """

        Parameters
        ----------
        x:
        pref: must have dimensions: [batch_size, p_dim]

        Returns
        -------

        """
        if pref is not None:
            x = torch.cat([x, pref], dim=1)
        return self.model.forward(x)
