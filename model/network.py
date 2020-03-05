import torch
import torch.nn as nn
import numpy as np


class ResidualFrame(object):

    def __init__(self, eps=1.0):
        super(ResidualFrame, self).__init__()

        self.eps = eps
        self.log_255 = np.float32(2 * np.log(255.0))
        self.log_max = np.float32(self.log_255 - np.log(self.eps))

    def __call__(self, x, y):

        d = torch.pow(255.0 * (x - y), 2)
        residual = self.log_255 - torch.log(d + self.eps)
        residual = residual / self.log_max

        return residual


class DownsampleConv3D(nn.Module):
    r"""
    Downsample by 2 over an input signal composed of several input planes with distinct spatial and time axes, by performing 3D convolution over the spatiotemporal axes

    args:
        in_channels (int): number of channels in the input tensor
        out_channels (int): number of channels produced by the convolution
        kernel_size (int or tuple): size of the convolution kernel
        stride (int or tuple): stride
        padding (int or tuple): zero-padding
        bias (bool, optional): whether to add a learnable bias
    """

    def __init__(self, in_channels, out_channels, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2), dilation=1, groups=1, bias=False):
        super(DownsampleConv3D, self).__init__()

        k = np.float32([1, 4, 6, 4, 1])
        k = np.outer(k, k)
        k5x5 = (k/k.sum()).reshape((1, 1, 1, 5, 5))

        conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        with torch.no_grad():
            conv1.weight = nn.Parameter(torch.from_numpy(k5x5))

        self.conv1 = conv1

    def forward(self, x):

        x = self.conv1(x)

        return x


class UpsampleConv3D(nn.Module):
    r"""
    Upsample by 2 over an input signal composed of several input planes with distinct spatial and time axes, by performing 3D convolution over the spatiotemporal axes

    rrgs:
        in_channels (int): number of channels in the input tensor
        out_channels (int): number of channels produced by the convolution
        kernel_size (int or tuple): size of the convolution kernel
        stride (int or tuple): stride
        padding (int or tuple): zero-padding
        bias (bool, optional): whether to add a learnable bias
    """

    def __init__(self, in_channels, out_channels, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2), dilation=1, groups=1, bias=False):
        super(UpsampleConv3D, self).__init__()

        k = np.float32([1, 4, 6, 4, 1])
        k = np.outer(k, k)
        k5x5 = (k/k.sum()).reshape((1, 1, 1, 5, 5))

        conv1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, output_padding=(0, 1, 1), bias=bias)

        with torch.no_grad():
            conv1.weight = nn.Parameter(torch.from_numpy(k5x5))

        self.conv1 = conv1

    def forward(self, x):

        x = self.conv1(x)

        return x


class SpatialConv3D(nn.Module):
    r"""
    Apply 3D conv. over an input signal composed of several input planes with distinct spatial and time axes, by performing 3D convolution over the spatiotemporal axes

    rrgs:
        in_channels (int): number of channels in the input tensor
        out_channels (int): number of channels produced by the convolution
        kernel_size (int or tuple): size of the convolution kernel
        stride (int or tuple): stride
        padding (int or tuple): zero-padding
    """

    def __init__(self, in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)):
        super(SpatialConv3D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size, stride, padding)
        self.reLu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(16, out_channels, kernel_size, stride, padding)
        self.reLu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.reLu1(x)
        x = self.conv2(x)
        x = self.reLu2(x)

        return x


class SpatialTemporalConv3D(nn.Module):
    r"""
    Apply 3D conv. over an input signal composed of several input planes with distinct spatial and time axes, by performing 3D convolution over the spatiotemporal axes

    args:
        in_channels (int): number of channels in the input tensor
        out_channels (int): number of channels produced by the convolution
        kernel_size (int or tuple): size of the convolution kernel
        stride (int or tuple): stride
        padding (int or tuple): zero-padding
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SpatialTemporalConv3D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size, stride, padding)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(64, 64, kernel_size, stride, padding)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv3d(64, 32, kernel_size, stride, padding)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.conv4 = nn.Conv3d(32, out_channels, kernel_size, stride, padding)
        self.relu4 = nn.LeakyReLU(inplace=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)

        return x


class C3DVQANet(nn.Module):

    def __init__(self):
        super(C3DVQANet, self).__init__()

        self.diff = ResidualFrame(eps=1.0)

        self.conv1_1 = DownsampleConv3D(1, 1)
        self.conv1_2 = UpsampleConv3D(1, 1)

        self.conv2_1 = SpatialConv3D(1, 16)
        self.conv2_2 = SpatialConv3D(1, 16)

        self.conv3 = SpatialTemporalConv3D(32, 1)

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.fc1 = nn.Linear(1, 4)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Linear(4, 1)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, ref, dis):

        err1 = self.diff(ref, dis)

        err2 = self.conv1_1(err1)    # 112x112 -> 56x56
        err2 = self.conv1_1(err2)    # 56x56 -> 28x28

        err3 = self.conv2_1(err1)    # 112x112 -> 56x56

        lo = dis
        for i in range(3):
            lo = self.conv1_1(lo)
        for i in range(3):
            lo = self.conv1_2(lo)
        dis = dis - lo

        dis = self.conv2_2(dis)    # 112x112 -> 56x56

        sens = torch.cat([dis, err3], dim=1)
        sens = self.conv3(sens)

        res = err2 * sens
        res = res[:, :, :, 4:-4, 4:-4]
        res = self.pool(res)

        res = self.fc1(res)
        res = self.relu1(res)
        res = self.fc2(res)
        res = self.relu2(res)
        res = torch.squeeze(res)

        return res
