import torch
import torch.nn as nn
from typing import Optional, Callable, Union, Tuple, Any
import torch
from torch import nn, Tensor
import numpy as np
from typing import Optional
import math
from torch import nn

def makeDivisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
def callMethod(self, ElementName):
    return getattr(self, ElementName)
def setMethod(self, ElementName, ElementValue):
    return setattr(self, ElementName, ElementValue)
def shuffleTensor(Feature: Tensor, Mode: int=1) -> Tensor:
    if isinstance(Feature, Tensor):
        Feature = [Feature]
    Indexs = None
    Output = []
    for f in Feature:
        B, C, H, W = f.shape
        if Mode == 1:
            f = f.flatten(2)
            if Indexs is None:
                Indexs = torch.randperm(f.shape[-1], device=f.device)
            f = f[:, :, Indexs.to(f.device)]
            f = f.reshape(B, C, H, W)
        else:
            if Indexs is None:
                Indexs = [torch.randperm(H, device=f.device),
                          torch.randperm(W, device=f.device)]
            f = f[:, :, Indexs[0].to(f.device)]
            f = f[:, :, :, Indexs[1].to(f.device)]
        Output.append(f)
    return Output
class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size: int or tuple=1):
        super(AdaptiveAvgPool2d, self).__init__(output_size=output_size)

    def profileModule(self, Input: Tensor):
        Output = self.forward(Input)
        return Output, 0.0, 0.0

class AdaptiveMaxPool2d(nn.AdaptiveMaxPool2d):
    def __init__(self, output_size: int or tuple=1):
        super(AdaptiveMaxPool2d, self).__init__(output_size=output_size)

    def profileModule(self, Input: Tensor):
        Output = self.forward(Input)
        return Output, 0.0, 0.0
class BaseConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: Optional[int] = 1,
            padding: Optional[int] = None,
            groups: Optional[int] = 1,
            bias: Optional[bool] = None,
            BNorm: bool = False,
            ActLayer: Optional[Callable[..., nn.Module]] = None,
            dilation: int = 1,
            Momentum: Optional[float] = 0.1,
            **kwargs: Any
    ) -> None:
        super(BaseConv2d, self).__init__()
        if padding is None:
            padding = int((kernel_size - 1) // 2 * dilation)

        if bias is None:
            bias = not BNorm

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias

        self.Conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, groups, bias, **kwargs)

        self.Bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=Momentum) if BNorm else nn.Identity()

        if ActLayer is not None:
            if isinstance(list(ActLayer().named_modules())[0][1], nn.Sigmoid):
                self.Act = ActLayer()
            else:
                self.Act = ActLayer(inplace=True)
        else:
            self.Act = ActLayer

        self.apply(initWeight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.Conv(x)
        x = self.Bn(x)
        if self.Act is not None:
            x = self.Act(x)
        return x

NormLayerTuple = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.SyncBatchNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.GroupNorm,
    nn.BatchNorm3d,
)
def initWeight(Module):
    if Module is None:
        return
    elif isinstance(Module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(Module.weight, a=math.sqrt(5))
        if Module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(Module.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(Module.bias, -bound, bound)
    elif isinstance(Module, NormLayerTuple):
        if Module.weight is not None:
            nn.init.ones_(Module.weight)
        if Module.bias is not None:
            nn.init.zeros_(Module.bias)
    elif isinstance(Module, nn.Linear):
        nn.init.kaiming_uniform_(Module.weight, a=math.sqrt(5))
        if Module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(Module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(Module.bias, -bound, bound)
    elif isinstance(Module, (nn.Sequential, nn.ModuleList)):
        for m in Module:
            initWeight(m)
    elif list(Module.children()):
        for m in Module.children():
            initWeight(m)
class Attention(nn.Module):
    def __init__(
            self,
            InChannels: int,
            HidChannels: int = None,
            SqueezeFactor: int = 4,
            PoolRes: list = [1, 2, 3],
            Act: Callable[..., nn.Module] = nn.ReLU,
            ScaleAct: Callable[..., nn.Module] = nn.Sigmoid,
            MoCOrder: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__()
        if HidChannels is None:
            HidChannels = max(makeDivisible(InChannels // SqueezeFactor, 8), 32)

        AllPoolRes = PoolRes + [1] if 1 not in PoolRes else PoolRes
        for k in AllPoolRes:
            Pooling = AdaptiveAvgPool2d(k)
            setMethod(self, 'Pool%d' % k, Pooling)

        self.SELayer = nn.Sequential(
            BaseConv2d(InChannels, HidChannels, 1, ActLayer=Act),
            BaseConv2d(HidChannels, InChannels, 1, ActLayer=ScaleAct),
        )

        self.PoolRes = PoolRes
        self.MoCOrder = MoCOrder

    def RandomSample(self, x: Tensor) -> Tensor:
        if self.training:
            PoolKeep = np.random.choice(self.PoolRes)
            x1 = shuffleTensor(x)[0] if self.MoCOrder else x
            AttnMap: Tensor = callMethod(self, 'Pool%d' % PoolKeep)(x1)
            if AttnMap.shape[-1] > 1:
                AttnMap = AttnMap.flatten(2)
                AttnMap = AttnMap[:, :, torch.randperm(AttnMap.shape[-1])[0]]
                AttnMap = AttnMap[:, :, None, None]  # squeeze twice
        else:
            AttnMap: Tensor = callMethod(self, 'Pool%d' % 1)(x)

        return AttnMap

    def forward(self, x: Tensor) -> Tensor:
        AttnMap = self.RandomSample(x)
        return x * self.SELayer(AttnMap)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x
class GLFA(nn.Module):
    def __init__(self, in_channels):
        super(GLFA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, padding=1, kernel_size=3, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, padding=2, kernel_size=3, dilation=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, padding=3, kernel_size=3, dilation=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, padding=4, kernel_size=3, dilation=4),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.mca = Attention(InChannels=in_channels, HidChannels=16)
    def forward(self, x):
        d = x
        c1 = self.conv_1(x)
        c2 = self.conv_2(x)
        c3 = self.conv_3(x)
        c4 = self.conv_4(x)
        cat = torch.cat([c1, c2, c3, c4], dim=1)
        cat = channel_shuffle(cat, groups=4)
        M= self.fuse(cat)  #
        O = self.mca(M)
        return O + d
