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

import sys
sys.path.append('/paddle/2d/PaddleClas')
from ppcls.arch.backbone.base.theseus_layer import TheseusLayer
from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

__all__ = ['CSPResNetB']


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
            bias_attr=False)

        self.bn = nn.BatchNorm2D(ch_out)
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            if self.act == 'leaky_relu':
                x = F.leaky_relu(x, 0.01)
            elif self.act == 'mish':
                x = mish(x)
            elif self.act == 'silu':
                x = x * F.sigmoid(x)
            else:
                x = getattr(F, self.act)(x)

        return x


class RepVggBlock(TheseusLayer):
    def __init__(self, ch_in, ch_out, act='relu'):
        super(RepVggBlock, self).__init__()
        self.conv1 = ConvBNLayer(
            ch_in, ch_out, 3, stride=1, padding=1, act=None)
        self.conv2 = ConvBNLayer(
            ch_in, ch_out, 1, stride=1, padding=0, act=None)
        self.act = act

    def forward(self, x):
        y = self.conv1(x) + self.conv2(x)
        if self.act == 'leaky_relu':
            y = F.leaky_relu(y, 0.01)
        elif self.act == 'mish':
            y = mish(y)
        elif self.act == 'silu':
            y = y * F.sigmoid(y)
        else:
            y = getattr(F, self.act)(y)
        return y


class BasicBlock(TheseusLayer):
    def __init__(self, ch_in, ch_out, act='relu', shortcut=True):
        super(BasicBlock, self).__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(
            ch_in, ch_out, 3, stride=1, padding=1, act=act)
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

    def __init__(self, channels, act='hardsigmoid'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2D(channels, channels, kernel_size=1, padding=0)
        self.act = act

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * getattr(F, self.act)(x_se)


class CSPResStage(TheseusLayer):
    def __init__(self,
                 block_fn,
                 ch_in,
                 ch_out,
                 n,
                 stride,
                 act='relu',
                 attn='eca'):
        super(CSPResStage, self).__init__()

        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.conv_down = ConvBNLayer(
                ch_in, ch_mid, 3, stride=2, padding=1, act=act)
        else:
            self.conv_down = None
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.blocks = nn.Sequential(* [
            block_fn(
                ch_mid // 2, ch_mid // 2, act='leaky_relu', shortcut=True)
            for i in range(n)
        ])
        if attn:
            self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid')
        else:
            self.attn = None

        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = paddle.concat([y1, y2], axis=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y


class CSPResNet(TheseusLayer):
    def __init__(self,
                 block_fn,
                 layers=[3, 6, 6, 3],
                 channels=[64, 128, 256, 512, 1024],
                 act='leaky_relu',
                 class_num=1000,
                 depth_wise=False):
        super(CSPResNet, self).__init__()

        self.class_num = class_num
        self.stem = nn.Sequential(
            ('conv1', ConvBNLayer(
                3, channels[0] // 2, 3, stride=2, padding=1,
                act=act)), ('conv2', ConvBNLayer(
                    channels[0] // 2,
                    channels[0],
                    3,
                    stride=1,
                    padding=1,
                    act=act)))

        n = len(channels) - 1
        self.stages = nn.Sequential(* [(str(i), CSPResStage(
            block_fn, channels[i], channels[i + 1], layers[i], 2, act=act))
                                       for i in range(n)])

        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.flatten = nn.Flatten()
        stdv = 1.0 / math.sqrt(channels[-1] * 1.0)
        self.fc = nn.Linear(
            channels[-1],
            self.class_num,
            weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)))

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


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
    depth_multiple = 1.
    width_multiple = 1.
    net = CSPResNet(
        BasicBlock, [int(n * depth_multiple) for n in [3, 6, 6, 3]],
        channels=[int(c * width_multiple) for c in [64, 128, 256, 512, 1024]],
        depth_wise=False)
    paddle.save(net.state_dict(), 'cspresnetb50_silu.pdparams')
    # p = 0
    # for k, v in net.state_dict().items():
    #     print(k, v.shape)
    #     p += np.prod(v.shape)

    # print('CSPResNet parameters: ', p)

    # x = paddle.randn((2, 3, 256, 256))
    # from paddle import jit
    # from paddle.static import InputSpec
    # net = jit.to_static(
    #     net, input_spec=[InputSpec(
    #         shape=[None, 3, 256, 256], name='x')])
    # jit.save(net, '/paddle/2d/PaddleClas/inference/cspresnet')
