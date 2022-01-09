from models.utils import nan_assert
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torch import _VF

from models.backbone.res1d import resnet18, resnet34
from models.backbone.tcn import MultiscaleMultibranchTCN

import math

class AudioCNN(nn.Module):
    def __init__(self, resnet_type):
        super(AudioCNN, self).__init__()
        # frontend1D
        self.frontend1D = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38, bias=False),
            nn.BatchNorm1d(64),
            nn.PReLU(num_parameters=64)
            )

        self.resnet = eval(resnet_type)()
        self.dropout = nn.Dropout(p=0.5)

        # initialize
        self._initialize_weights()

    def forward(self, x):
        x = self.frontend1D(x)

        x = self.resnet.forward_without_conv(x).transpose(1, 2)

        return x

    def block_forward_para(self, x, params, base):
        x = F.conv1d(x, params[base + '.frontend1D.0.weight'], bias=None, stride=(4,), padding=(38,))
        x = F.batch_norm(x, weight=params[base + '.frontend1D.1.weight'], bias=params[base + '.frontend1D.1.bias'],
            running_mean=self._modules['frontend1D']._modules['1'].running_mean,
            running_var=self._modules['frontend1D']._modules['1'].running_var,
            training=self.training)
        x = F.prelu(x, params[base + '.frontend1D.2.weight'])
        return self.resnet.block_forward_para_without_conv(x, params, base).transpose(1, 2)

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

class Conv1dResNet(nn.Module):
    def __init__(self, resnet_type, backend_type):
        super(Conv1dResNet, self).__init__()
        self.audio_cnn = AudioCNN(resnet_type)

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
    
    def gru_block_forward_para(self, input, params, base):
        _flat_weights = [params[i] for i in params.keys() if base in i]
        max_batch_size = input.size(0) if self.gru.batch_first else input.size(1)
        unsorted_indices = None

        num_directions = 2 if self.gru.bidirectional else 1
        hx = torch.zeros(self.gru.num_layers * num_directions,
                            max_batch_size, self.gru.hidden_size,
                            dtype=input.dtype, device=input.device)

        result = _VF.gru(input, hx, _flat_weights, self.gru.bias, self.gru.num_layers,
                            self.gru.dropout, self.gru.training, self.gru.bidirectional, self.gru.batch_first)

        output = result[0]
        hidden = result[1]

        return output, self.gru.permute_hidden(hidden, unsorted_indices)

    def lstm_block_forward_para(self, input, params, base):
        _flat_weights = [params[i] for i in params.keys() if base in i]
        max_batch_size = input.size(0) if self.gru.batch_first else input.size(1)
        unsorted_indices = None

        num_directions = 2 if self.gru.bidirectional else 1
        real_hidden_size = self.gru.proj_size if self.gru.proj_size > 0 else self.gru.hidden_size
        h_zeros = torch.zeros(self.gru.num_layers * num_directions,
                                max_batch_size, real_hidden_size,
                                dtype=input.dtype, device=input.device)
        c_zeros = torch.zeros(self.gru.num_layers * num_directions,
                                max_batch_size, self.gru.hidden_size,
                                dtype=input.dtype, device=input.device)
        hx = (h_zeros, c_zeros)

        result = _VF.lstm(input, hx, _flat_weights, self.gru.bias, self.gru.num_layers,
                            self.gru.dropout, self.gru.training, self.gru.bidirectional, self.gru.batch_first)
        output = result[0]
        hidden = result[1:]
        return output, self.gru.permute_hidden(hidden, unsorted_indices)

    def block_forward_para(self, x, params, base, relu=False):
        if self.backend_type in ['GRU', 'LSTM']:
            self.gru.flatten_parameters()

        if self.training:
            with autocast():
                f_v = self.audio_cnn.block_forward_para(x.unsqueeze(1), params, base + '.audio_cnn')
                f_v = F.dropout(f_v, p=0.3, training=self.training, inplace=True)
        else:
            f_v = self.audio_cnn.block_forward_para(x.unsqueeze(1), params, base + '.audio_cnn')
            f_v = F.dropout(f_v, p=0.3, training=self.training, inplace=True)
        f_v = f_v.float()

        # f_v: [batch_size, 29, 512]

        if self.backend_type == 'GRU':
            h, _ = self.gru_block_forward_para(f_v, params, base + '.gru')
        elif self.backend_type == 'LSTM':
            h, _ = self.lstm_block_forward_para(f_v, params, base + '.gru')

        # h = self.dropout(h)
        if relu:
            h = self.relu(h)

        pred = F.linear(h.contiguous().view(h.shape[0], -1), weight=params[base + '.fc.weight'], bias=params[base + '.fc.bias'])
        return pred


    def forward(self, x, relu=False):
        # x: [batch_size, 29, 1, 88, 88]
        if self.backend_type in ['GRU', 'LSTM']:
            self.gru.flatten_parameters()

        if self.training:
            with autocast():
                f_v = self.audio_cnn(x.unsqueeze(1))
                f_v = self.dropout(f_v)
            f_v = f_v.float()
        else:
            f_v = self.audio_cnn(x.unsqueeze(1))
            f_v = self.dropout(f_v)
            f_v = f_v.float()

        # f_v: [batch_size, 29, 512]

        h, _ = self.gru(f_v) # h: [batch_size, 29, 2048]

        # h = self.dropout(h)
        if relu:
            h = self.relu(h)
        return h