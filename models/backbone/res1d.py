import math
import torch.nn as nn
import torch.nn.functional as F

import pdb

from collections import OrderedDict


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def downsample_basic_block( inplanes, outplanes, stride ):
    return  nn.Sequential(
                nn.Conv1d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outplanes),
            )

def downsample_basic_block_v2( inplanes, outplanes, stride ):
    return  nn.Sequential(
                nn.AvgPool1d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv1d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(outplanes),
            )



class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type = 'relu'):
        super(BasicBlock1D, self).__init__()

        assert relu_type in ['relu','prelu']

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)

        # type of ReLU is an input option
        if relu_type == 'relu':
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == 'prelu':
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        else:
            raise Exception('relu type not implemented')
        # --------

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
 
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out

    def block_forward_para(self, x, params, base, mode, modules, downsample=False):
        residual = x

        out = F.conv1d(x, params[base + 'conv1.weight'], stride=(self.stride,), padding=(1,))

        out = F.batch_norm(out, weight=params[base + 'bn1.weight'], bias=params[base + 'bn1.bias'],
            running_mean=modules['bn1'].running_mean,
            running_var=modules['bn1'].running_var, training = mode)
        out = self.relu1(out)

        out = F.conv1d(out, params[base + 'conv2.weight'], stride=(1,), padding=(1,))
        out = F.batch_norm(out, weight=params[base + 'bn2.weight'], bias=params[base + 'bn2.bias'],
            running_mean=modules['bn2'].running_mean,
            running_var=modules['bn2'].running_var, training = mode)

        if downsample is True:
            residual = F.conv1d(x, params[base + 'downsample.0.weight'], stride=(self.stride,))
            residual = F.batch_norm(residual, weight=params[base + 'downsample.1.weight'], 
                                    bias=params[base + 'downsample.1.bias'],
                                    running_mean=modules['downsample']._modules['1'].running_mean,
                                    running_var=modules['downsample']._modules['1'].running_var, training = mode)

        out += residual

        out = self.relu2(out)


        return out


class ResNet1D(nn.Module):
    def __init__(self, block=BasicBlock1D, layers=[2, 2, 2, 2], relu_type='relu'):
        super(ResNet1D, self).__init__()
        self.inplanes = 64
        self.relu_type = relu_type
        self.downsample_block = downsample_basic_block

        # self.conv1 = nn.Conv1d(1, self.inplanes, kernel_size=80, stride=4, padding=38,
        #                        bias=False)
        # self.bn1 = nn.BatchNorm1d(self.inplanes)
        # type of ReLU is an input option
        # if relu_type == 'relu':
        #     self.relu = nn.ReLU(inplace=True)
        # elif relu_type == 'prelu':
        #     self.relu = nn.PReLU(num_parameters=self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # For LRW, we downsample the sampling rate to 25fps
        self.avgpool = nn.AvgPool1d(kernel_size=21, padding=1)
        '''
        # The following pooling setting is the general configuration
        self.avgpool = nn.AvgPool1d(kernel_size=20, stride=20)
        '''

        # default init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block( inplanes = self.inplanes, 
                                                 outplanes = planes * block.expansion, 
                                                 stride = stride )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, relu_type = self.relu_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type = self.relu_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        raise NotImplementedError
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.avgpool(x)

        # return x

    def forward_without_conv(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x

    def block_forward_para_without_conv(self, x, params, base, embedding=False):
        x = self.layer1[0].block_forward_para(x, params, base + '.resnet.layer1.0.', self.training, self._modules['layer1']._modules['0']._modules, downsample=False)
        x = self.layer1[1].block_forward_para(x, params, base + '.resnet.layer1.1.', self.training, self._modules['layer1']._modules['1']._modules, downsample=False)
        x = self.layer2[0].block_forward_para(x, params, base + '.resnet.layer2.0.', self.training, self._modules['layer2']._modules['0']._modules, downsample=True)
        x = self.layer2[1].block_forward_para(x, params, base + '.resnet.layer2.1.', self.training, self._modules['layer2']._modules['1']._modules, downsample=False)
        x = self.layer3[0].block_forward_para(x, params, base + '.resnet.layer3.0.', self.training, self._modules['layer3']._modules['0']._modules, downsample=True)
        x = self.layer3[1].block_forward_para(x, params, base + '.resnet.layer3.1.', self.training, self._modules['layer3']._modules['1']._modules, downsample=False)
        x = self.layer4[0].block_forward_para(x, params, base + '.resnet.layer4.0.', self.training, self._modules['layer4']._modules['0']._modules, downsample=True)
        x = self.layer4[1].block_forward_para(x, params, base + '.resnet.layer4.1.', self.training, self._modules['layer4']._modules['1']._modules, downsample=False)

        x = self.avgpool(x)

        return x


def resnet10(**kwargs):
    """Constructs a ResNet-10 model.
    """
    model = ResNet1D(BasicBlock1D, [1, 1, 1, 1], **kwargs)
    return model

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet1D(BasicBlock1D, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet1D(BasicBlock1D, [3, 4, 6, 3], **kwargs)
    return model