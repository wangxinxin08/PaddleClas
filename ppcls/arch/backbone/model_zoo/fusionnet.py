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

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm2D, Linear
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform
import math

# import sys
# sys.path.append('/paddle/2d/PaddleClas')
from ppcls.arch.backbone.base.theseus_layer import TheseusLayer
from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

__all__ = ['CSPFusionNet', 'CSPFusionNetV1']


class ConvBNLayer(TheseusLayer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act='',
                 data_format="NCHW"):
        super(ConvBNLayer, self).__init__()
        self.act = act
        self.conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias_attr=False,
            data_format=data_format)
        self.bn = BatchNorm2D(num_filters, data_format=data_format)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act == 'leaky_relu':
            x = F.leaky_relu(x, 0.1)
        else:
            x = getattr(F, self.act)(x)
        return x


class FusionBlock(TheseusLayer):
    def __init__(self, ch_in, ch_out, ratio=4, act='relu', shortcut=True):
        super(FusionBlock, self).__init__()

        ch_mid = ch_out // ratio
        self.conv1 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid * 2, 3, act=act)
        self.conv3 = ConvBNLayer(ch_mid * 2, ch_mid, 3, act=act)
        self.conv4 = ConvBNLayer(ch_out, ch_out, 1, act=act)
        self.shortcut = shortcut
        if self.shortcut and ch_in != ch_out:
            self.short = ConvBNLayer(ch_in, ch_out, 1, act=act)
        else:
            self.short = None

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y = paddle.concat([y1, y2, y3], axis=1)
        y = self.conv4(y)
        if self.shortcut:
            if self.short is not None:
                return paddle.add(y, self.short(x))
            else:
                return paddle.add(y, x)

        return y


class FusionStage(TheseusLayer):
    def __init__(self, ch_in, ch_out, n, stride, act='relu'):
        super(FusionStage, self).__init__()

        if stride == 2:
            self.conv_down = ConvBNLayer(ch_in, ch_out, 3, stride=2, act=act)
            ch_in = ch_out
        else:
            self.conv_down = None

        self.blocks = nn.Sequential(* [
            FusionBlock(
                ch_in if i == 0 else ch_out, ch_out, act=act, shortcut=True)
            for i in range(n)
        ])

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)

        x = self.blocks(x)
        return x


class FusionConv(TheseusLayer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 ratio=2,
                 act='relu',
                 shortcut=True,
                 depth_wise=False):
        super(FusionConv, self).__init__()
        ch_mid = ch_out // ratio
        self.conv1 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.conv2 = ConvBNLayer(
            ch_mid, ch_mid, 3, act=act, groups=ch_mid if depth_wise else 1)
        self.conv3 = ConvBNLayer(
            ch_mid, ch_mid, 3, act=act, groups=ch_mid if depth_wise else 1)
        self.conv4 = ConvBNLayer(ch_out, ch_out, 1, act=act)
        self.shortcut = shortcut
        if self.shortcut and ch_in != ch_out:
            self.short = ConvBNLayer(ch_in, ch_out, 1, act=act)
        else:
            self.short = None

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y = paddle.concat([y2, y3], axis=1)
        y = self.conv4(y)
        if self.shortcut:
            if self.short is not None:
                return paddle.add(y, self.short(x))
            else:
                return paddle.add(y, x)

        return y


class CSPFusionStage(TheseusLayer):
    def __init__(self, ch_in, ch_out, n, stride, act='relu', depth_wise=False):
        super(CSPFusionStage, self).__init__()

        if stride == 2:
            self.conv_down = ConvBNLayer(ch_in, ch_out, 3, stride=2, act=act)
            ch_in = ch_out
        else:
            self.conv_down = None

        ch_mid = ch_out // 2
        self.conv1 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.conv2 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.blocks = nn.Sequential(* [
            FusionConv(
                ch_mid, ch_mid, act=act, shortcut=True, depth_wise=depth_wise)
            for i in range(n)
        ])
        self.conv3 = ConvBNLayer(ch_mid * 2, ch_out, 1, act=act)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)

        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = paddle.concat([y1, y2], axis=1)
        y = self.conv3(y)
        return y


class FusionNet(TheseusLayer):
    def __init__(self,
                 layers,
                 channels=[64, 128, 256, 512, 1024],
                 act='relu',
                 return_idx=[0, 1, 2, 3]):
        super(FusionNet, self).__init__()

        self.stem = nn.Sequential(
            ConvBNLayer(
                3, 64, 3, stride=2, act=act),
            ConvBNLayer(
                64, 64, 3, stride=1, act=act),
            # ConvBNLayer(3, 128, 3, stride=2, act=act)
        )

        n = len(channels) - 1
        self.stages = nn.Sequential(* [
            FusionStage(
                channels[i], channels[i + 1], layers[i], 2, act=act)
            for i in range(n)
        ])

        self.return_idx = return_idx

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.return_idx:
                outs.append(x)

        return outs


class CSPFusionNet(TheseusLayer):
    def __init__(self,
                 layers,
                 channels=[32, 64, 128, 256, 512, 1024],
                 act='leaky_relu',
                 return_idx=[0, 1, 2, 3, 4],
                 class_num=1000,
                 depth_wise=False):
        super(CSPFusionNet, self).__init__()

        self.class_num = class_num
        self.stem = nn.Sequential(
            ConvBNLayer(
                3, channels[0], 3, stride=1, act=act),
            # ConvBNLayer(3, 128, 3, stride=2, act=act)
        )

        n = len(channels) - 1
        self.stages = nn.Sequential(* [
            CSPFusionStage(
                channels[i],
                channels[i + 1],
                layers[i],
                2,
                act=act,
                depth_wise=depth_wise) for i in range(n)
        ])

        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.flatten = nn.Flatten()
        stdv = 1.0 / math.sqrt(channels[-1] * 1.0)
        self.fc = nn.Linear(
            channels[-1],
            self.class_num,
            weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)))

        self.return_idx = return_idx

    def forward(self, x):
        x = self.stem(x)
        # outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            # print(x.shape)
            # if i in self.return_idx:
            #     outs.append(x)

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


def CSPFusionNetV1(layers, pretrained=False, use_ssld=False, **kwargs):
    model = CSPFusionNet(layers, **kwargs)
    return model


if __name__ == '__main__':
    import sys
    sys.path.append('/paddle/2d/PaddleClas')
    net = CSPFusionNet(
        [1, 3, 12, 12, 6],
        channels=[32, 64, 128, 256, 512, 1024],
        depth_wise=False)
    p = 0
    for k, v in net.state_dict().items():
        p += np.prod(v.shape)

    print(p)

    # print('CSPFusionNet parameters: ', p)
    # x = paddle.randn((2, 3, 256, 256))
    # from paddle import jit
    # from paddle.static import InputSpec
    # net = jit.to_static(
    #     net, input_spec=[InputSpec(
    #         shape=[None, 3, 256, 256], name='x')])
    # jit.save(net, '/paddle/2d/PaddleClas/inference/cspfusionnet')
