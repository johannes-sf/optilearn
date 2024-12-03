import torch.nn as nn
import torch.nn.functional as F
from optilearn.models.utils.generator_utils import get_func, init_layers


class BasicConv2dBlock(nn.Module):
    """

    """
    def __init__(self, in_channels, filters, kernel_size, stride=1, padding=0, bias=True, act='relu', pooling=1,
                 batch_norm=False):
        super(BasicConv2dBlock, self).__init__()
        self.pooling = pooling
        self.batch_norm = batch_norm
        self.conv = nn.Conv2d(in_channels,
                              filters,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.act = get_func(act)
        if batch_norm:
            self.bn = nn.BatchNorm2d(filters, eps=0.001)

        self.mp = nn.MaxPool2d(pooling, pooling)

        init_layers(self.modules())
        self.out_channels = filters


    def forward(self, x):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.act(x)
        x = self.mp(x)
        return x


class ResBlock(nn.Module):
    """

    """
    def __init__(self, in_channels, filters, kernel_size, act):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               filters,
                               kernel_size=kernel_size,
                               padding=int((kernel_size - 1) / 2),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(filters, eps=0.001)
        self.act1 = get_func(act)
        self.conv2 = nn.Conv2d(filters,
                               filters,
                               kernel_size=kernel_size,
                               padding=int((kernel_size - 1) / 2),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(filters, eps=0.001)
        self.act2 = get_func(act)

        self.skip_conv = BasicConv2dBlock(in_channels, filters, kernel_size=1)

        init_layers(self.modules())
        self.out_channels = filters

    def forward(self, x):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.act1(z)

        z = self.conv2(z)
        z = self.bn2(z)
        z = z + self.skip_conv(x)
        z = self.act2(z)
        return z


class ReductionBlock(nn.Module):
    """

    """
    def __init__(self, in_channels, out_channels):
        super(ReductionBlock, self).__init__()
        self.conv0 = BasicConv2dBlock(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2dBlock(128, out_channels, kernel_size=5)
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        self.out_channels = out_channels


    def forward(self, x):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x