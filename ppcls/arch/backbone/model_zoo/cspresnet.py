# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

from __future__ import absolute_import, division, print_function

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import KaimingNormal
from paddle.nn import Conv2D, BatchNorm2D, Linear
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform
import math

# import sys
# sys.path.append('/paddle/2d/PaddleClas')
from ppcls.arch.backbone.base.theseus_layer import TheseusLayer
from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

__all__ = ['CSPResNetB', 'CSPResNetW']


class ConvBNLayer(TheseusLayer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2D(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(initializer=KaimingNormal()),
            bias_attr=False)

        self.bn = nn.BatchNorm2D(ch_out)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            if self.act == 'leaky_relu':
                x = F.leaky_relu(x, 0.1)
            elif self.act == 'mish':
                x = mish(x)
            else:
                x = getattr(F, self.act)(x)

        return x


class RepVggBlock(TheseusLayer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 act='relu'):
        super(RepVggBlock, self).__init__()
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=None)
        self.conv2 = ConvBNLayer(ch_in, ch_out, 1, stride=1, padding=0, act=None)
        self.act = act


    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        if self.act == 'leaky':
            y = F.leaky_relu(y, 0.1)
        elif self.act == 'mish':
            y = mish(y)
        else:
            y = getattr(F, self.act)(y)
        return y
    
    # def eval(self):
    #     if not hasattr(self, 'conv'):
    #         self.conv = nn.Conv2D(
    #             in_channels=self.in_channels,
    #             out_channels=self.out_channels,
    #             kernel_size=self.kernel_size,
    #             stride=self.stride,
    #             padding=self.padding,
    #             dilation=self.dilation,
    #             groups=self.groups,
    #             padding_mode=self.padding_mode)
    #     self.training = False
    #     kernel, bias = self.get_equivalent_kernel_bias()
    #     self.conv.weight.set_value(kernel)
    #     self.conv.bias.set_value(bias)
    #     for layer in self.sublayers():
    #         layer.eval()

    # def get_equivalent_kernel_bias(self):
    #     kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
    #     kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
    #     return kernel3x3 + self._pad_1x1_to_3x3_tensor(
    #         kernel1x1), bias3x3 + bias1x1

    # def _pad_1x1_to_3x3_tensor(self, kernel1x1):
    #     if kernel1x1 is None:
    #         return 0
    #     else:
    #         return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    # def _fuse_bn_tensor(self, branch):
    #     if branch is None:
    #         return 0, 0
    #     kernel = branch.conv.weight
    #     running_mean = branch.bn._mean
    #     running_var = branch.bn._variance
    #     gamma = branch.bn.weight
    #     beta = branch.bn.bias
    #     eps = branch.bn._epsilon
    #     std = (running_var + eps).sqrt()
    #     t = (gamma / std).reshape((-1, 1, 1, 1))
    #     return kernel * t, beta - running_mean * gamma / std


class WideRepVggBlock(TheseusLayer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 act='relu'):
        super(WideRepVggBlock, self).__init__()
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=None)
        self.conv2 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=None)
        self.act = act


    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        if self.act == 'leaky':
            y = F.leaky_relu(y, 0.1)
        elif self.act == 'mish':
            y = mish(y)
        else:
            y = getattr(F, self.act)(y)
        return y
    
    # def eval(self):
    #     if not hasattr(self, 'conv'):
    #         self.conv = nn.Conv2D(
    #             in_channels=self.in_channels,
    #             out_channels=self.out_channels,
    #             kernel_size=self.kernel_size,
    #             stride=self.stride,
    #             padding=self.padding,
    #             dilation=self.dilation,
    #             groups=self.groups,
    #             padding_mode=self.padding_mode)
    #     self.training = False
    #     kernel, bias = self.get_equivalent_kernel_bias()
    #     self.conv.weight.set_value(kernel)
    #     self.conv.bias.set_value(bias)
    #     for layer in self.sublayers():
    #         layer.eval()

    # def get_equivalent_kernel_bias(self):
    #     kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
    #     kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
    #     return kernel3x3 + kernel1x1, bias3x3 + bias1x1

    # def _pad_1x1_to_3x3_tensor(self, kernel1x1):
    #     if kernel1x1 is None:
    #         return 0
    #     else:
    #         return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    # def _fuse_bn_tensor(self, branch):
    #     if branch is None:
    #         return 0, 0
    #     kernel = branch.conv.weight
    #     running_mean = branch.bn._mean
    #     running_var = branch.bn._variance
    #     gamma = branch.bn.weight
    #     beta = branch.bn.bias
    #     eps = branch.bn._epsilon
    #     std = (running_var + eps).sqrt()
    #     t = (gamma / std).reshape((-1, 1, 1, 1))
    #     return kernel * t, beta - running_mean * gamma / std


class WideBlock(TheseusLayer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 act='relu',
                 shortcut=True):
        super(WideBlock, self).__init__()
        assert ch_in == ch_out
        self.conv1 = WideRepVggBlock(ch_out, ch_out, act=act)
        self.conv2 = RepVggBlock(ch_out, ch_out, act=act)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return paddle.add(x, y)
        else:
            return y


class BasicBlock(TheseusLayer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 act='relu',
                 shortcut=True):
        super(BasicBlock, self).__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=act)
        self.conv2 = RepVggBlock(ch_out, ch_out, act=act)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return paddle.add(x, y)
        else:
            return y


class EffectiveSELayer(TheseusLayer):
    """ Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, act='sigmoid'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2D(channels, channels, kernel_size=1, padding=0)
        self.act = act

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * getattr(F, self.act)(x_se)


class CSPResStage(TheseusLayer):
    def __init__(self, block_fn, ch_in, ch_out, n, stride, act='relu', attn='eca'):
        super(CSPResStage, self).__init__()

        ch_mid = ch_in // 2
        self.conv1 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.conv2 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.blocks = nn.Sequential(* [
            block_fn(
                ch_mid, ch_mid, act=act, shortcut=True)
            for i in range(n)
        ])
        if attn:
            self.attn = EffectiveSELayer(ch_mid * 2, act='sigmoid')
        else:
            self.attn = None

        self.conv3 = ConvBNLayer(ch_mid * 2, ch_mid * 2, 1, act=act)
        if stride == 2:
            self.conv_down = ConvBNLayer(ch_mid * 2, ch_out, 3, stride=2, padding=1, act=act)
        else:
            self.conv_down = None

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = paddle.concat([y1, y2], axis=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        if self.conv_down is not None:
            y = self.conv_down(y)
        return y


class CSPResNet(TheseusLayer):
    def __init__(self,
                 block_fn,
                 layers=[2, 4, 6, 8],
                 channels=[64, 128, 256, 512, 1024],
                 act='leaky_relu',
                 class_num=1000,
                 depth_wise=False):
        super(CSPResNet, self).__init__()

        self.class_num = class_num
        self.stem = ConvBNLayer(3, channels[0], 3, stride=2, act=act)

        n = len(channels) - 1
        self.stages = nn.Sequential(* [
            CSPResStage(
                block_fn,
                channels[i],
                channels[i + 1],
                layers[i],
                2,
                act=act) for i in range(n)
        ])

        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.flatten = nn.Flatten()
        stdv = 1.0 / math.sqrt(channels[-1] * 1.0)
        self.fc = nn.Linear(
            channels[-1],
            self.class_num,
            weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)))

    def forward(self, x):
        x = self.stem(x)
        for idx, stage in enumerate(self.stages):
            x = stage(x)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


def CSPResNetW(pretrained=False, use_ssld=False, **kwargs):
    """
    CSPResNet
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ResNet101_vd` model depends on args.
    """
    model = CSPResNet(block_fn=WideBlock, **kwargs)
    return model


def CSPResNetB(pretrained=False, use_ssld=False, **kwargs):
    """
    CSPResNet
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `ResNet101_vd` model depends on args.
    """
    model = CSPResNet(block_fn=BasicBlock, **kwargs)
    return model


if __name__ == '__main__':
    width_multiple = 1
    depth_multiple = 1
    net = CSPResNet(
        WideBlock,
        [int(n * depth_multiple) for n in [2, 4, 8, 12]],
        channels=[int(c * width_multiple) for c in [64, 128, 256, 512, 1024]],
        depth_wise=False)
    p = 0
    for k, v in net.state_dict().items():
        print(k, v.shape)
        p += np.prod(v.shape)

    print('CSPResNet parameters: ', p)

    # x = paddle.randn((2, 3, 256, 256))
    # from paddle import jit
    # from paddle.static import InputSpec
    # net = jit.to_static(
    #     net, input_spec=[InputSpec(
    #         shape=[None, 3, 256, 256], name='x')])
    # jit.save(net, '/paddle/2d/PaddleClas/inference/cspresnet')
