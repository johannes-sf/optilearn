from collections import OrderedDict

from torch import nn

from optilearn.models.nn_generators.abstract_nn import AbstractNN
from optilearn.models.nn_generators.mlp_blocks import BasicMlpBlock


class MLP(AbstractNN):
    """ """

    def __init__(self, dim_in, dim_out, nn_config, device):
        super(MLP, self).__init__(dim_in, None, dim_out, nn_config)
        layers = OrderedDict()
        for i, layer in enumerate(nn_config.values()):
            layers[str(i)] = BasicMlpBlock(
                d_in=dim_in,
                size=layer["size"],
                act=layer["act"],
            )
            dim_in = layers[str(i)].out_dim

        layers["out"] = nn.Linear(dim_in, dim_out)
        self.model = nn.Sequential(layers)
        self.to(device)
