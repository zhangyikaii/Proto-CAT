import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler

from models.backbone.res18 import resnet18, resnet34
from models.backbone.tcn import MultiscaleMultibranchTCN
from models.utils import nan_assert

import math

class VideoCNN(nn.Module):
    def __init__(self, resnet_type):
        super(VideoCNN, self).__init__()
        # frontend3D
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            )

        # resnet
        self.resnet18 = eval(resnet_type)()
        self.dropout = nn.Dropout(p=0.5)

        # initialize
        self._initialize_weights()

    def forward(self, x):
        b = x.size(0)
        x = x.transpose(1, 2)
        x = self.frontend3D(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))
        x = self.resnet18.forward_without_conv(x)
        x = x.view(b, -1, 512)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Conv3dResNet(nn.Module):
    def __init__(self, resnet_type, backend_type):
        super(Conv3dResNet, self).__init__()
        self.video_cnn = VideoCNN(resnet_type)

        self.backend_type = backend_type
        if backend_type == 'GRU':
            self.gru = nn.GRU(512, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)
        elif backend_type == 'LSTM':
            self.gru = nn.LSTM(512, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)
        elif backend_type == 'MSTCN':
            tcn_options = {'num_layers': 3, 'kernel_size': [3, 5, 7, 9], 'dropout': 0.2, 'dwpw': False, 'width_mult': 2}
            self.gru = MultiscaleMultibranchTCN(
                input_size=512,
                num_channels=[256 * len(tcn_options['kernel_size']) * tcn_options['width_mult']] * tcn_options['num_layers'],
                num_classes=100,
                tcn_options=tcn_options,
                dropout=tcn_options['dropout'],
                relu_type='prelu',
                dwpw=tcn_options['dwpw']
                )
        else:
            raise NotImplementedError
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.LeakyReLU(0.5, inplace=True)
        # self.v_cls = nn.Linear(1024*2, 500)


    def forward(self, x, relu=False):
        # x: [batch_size, 29, 1, 88, 88]
        if self.backend_type in ['GRU', 'LSTM']:
            self.gru.flatten_parameters()

        if self.training:
            with autocast():
                f_v = self.video_cnn(x)
                f_v = self.dropout(f_v)
            f_v = f_v.float()
        else:
            f_v = self.video_cnn(x)
            f_v = self.dropout(f_v)
            f_v = f_v.float()

        # f_v: [batch_size, 29, 512]

        h, _ = self.gru(f_v) # h: [batch_size, 29, 2048]

        # h = self.dropout(h)
        if relu:
            h = self.relu(h)
        return h
