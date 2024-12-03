import torch
from torch import nn
from collections import OrderedDict
from copy import deepcopy
from optilearn.models.nn_generators.abstract_nn import AbstractNN
from optilearn.models.nn_generators.cnn_blocks import BasicConv2dBlock, ResBlock, ReductionBlock
from optilearn.models.nn_generators.inception_blocks import get_inception_kernel


class CNN(AbstractNN):
    """

    """
    def __init__(self, dim_in, nn_config, input_size, device):
        super(CNN, self).__init__(dim_in, None, None, nn_config)
        layers = OrderedDict()
        d_in = deepcopy(dim_in)
        for i, layer in enumerate(nn_config.values()):
            layers[str(i)] = (self._get_block(layer, d_in))
            d_in = layers[str(i)].out_channels
        layers['flatten'] = nn.Flatten()
        self.model = nn.Sequential(layers)
        self.dim_out = self._get_cnn_dim_out(self.model, input_size=input_size)
        self.to(device)

    @staticmethod
    def _get_block(layer, d_in):
        if layer['block'] == 'basic_cnn':
            return BasicConv2dBlock(d_in,
                                    layer['size'],
                                    kernel_size=layer['kernel'],
                                    stride=layer['stride'],
                                    padding=layer['padding'],
                                    act=layer['act'],
                                    bias=layer['bias'],
                                    batch_norm=layer['batch_norm'])
        elif layer['block'] == 'residual':
            return ResBlock(in_channels=d_in,
                            filters=layer['size'],
                            kernel_size=layer['kernel'],
                            act=layer['act'])
        elif layer['block'] == 'inception':
            return get_inception_kernel(kernel_type=layer['type'], in_channels=d_in, pool_feature=layer['pool_feature'],
                                        channels_7x7=layer['channels_7x7'])
        elif layer['block'] == 'cnn_reduction':
            return ReductionBlock(in_channels=d_in, out_channels=layer['size'])

    def _get_cnn_dim_out(self, cnn, input_size):
        x = torch.zeros(1,  self.dim_in, *input_size)
        with torch.no_grad():
            x = cnn.forward(x)
        return x.shape[-1]
