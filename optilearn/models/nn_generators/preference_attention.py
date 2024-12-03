import torch
from typing import Optional

from optilearn.models.nn_generators.abstract_nn import AbstractNN
from optilearn.models.nn_generators.mlp import MLP


class PreferenceAttention(AbstractNN):
    def __init__(self, dim_in, dim_p, dim_out, nn_config, device='cpu'):
        super(PreferenceAttention, self).__init__(dim_in, dim_out, dim_p, nn_config)
        nn_config_ = nn_config.copy()
        last_layer_key = max(nn_config_.keys())
        last_layer_size = nn_config_.pop(last_layer_key)['size']
        self.mlp = MLP(dim_in=dim_in, dim_out=last_layer_size, nn_config=nn_config_, device=device)
        self.attention = MLP(dim_in=dim_p, dim_out=last_layer_size, nn_config=nn_config_, device=device)
        self.out = MLP(dim_in=last_layer_size, dim_out=dim_out, nn_config=nn_config_, device=device)
        self.to(device)

    def forward(self, x, pref: Optional[torch.Tensor] = None):
        x = self.mlp.forward(x)
        w = 1 + self.attention.forward(pref)
        return self.out.forward(x * w)


if __name__ == '__main__':
    nn_cfg = {0: {'size': 128, 'act': 'lrelu'},
              1: {'size': 256, 'act': 'lrelu'},
              2: {'size': 64, 'act': 'lrelu'}}

    model = PreferenceAttention(5, 2, 4, nn_cfg)
    x = torch.randn(32, 5)
    pref = torch.randn(32, 2)
    model.forward(x, pref)
