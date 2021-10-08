import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

from models.metrics import pairwise_distances

class BidrectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        """Bidirectional LSTM used to generate fully conditional embeddings (FCE) of the support set as described
        in the Matching Networks paper.

        # Arguments
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            num_layers: Number of LSTM layers
        """
        super(BidrectionalLSTM, self).__init__()
        # Force input size and hidden size to be the same in order to implement
        # the skip connection as described in Appendix A.1 and A.2 of Matching Networks
        self.lstm = nn.LSTM(input_size=input_size,
                            num_layers=num_layers,
                            hidden_size=hidden_size,
                            batch_first=True,
                            bidirectional=True)

    # def cuda(self):
    #     super(BidrectionalLSTM, self).cuda()
    #     return self
    
    def forward(self, inputs):
        # Give None as initial state and Pytorch LSTM creates initial hidden states
        output, (hn, cn) = self.lstm(inputs.unsqueeze(0), None)
        output = output.squeeze(0)      

        forward_output = output[:, :self.lstm.hidden_size]
        backward_output = output[:, self.lstm.hidden_size:]

        # g(x_i, S) = h_forward_i + h_backward_i + g'(x_i) as written in Appendix A.2
        # AKA A skip connection between inputs and outputs is used
        output = forward_output + backward_output + inputs

        # Trick:
        output_norm = torch.norm(output, p=2, dim=1).unsqueeze(1).expand_as(output)
        output_normalized = output.div(output_norm + 0.00001)
        return output, output_normalized

class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(AttentionLSTM, self).__init__()
        self.num_layers = num_layers
        self.lstm_cell = nn.LSTMCell(
            input_size=input_size * 2,
            hidden_size=hidden_size
        ).to(torch.device('cuda'))
        self.input_size = input_size
        self.c_0 = Variable(torch.zeros(1, input_size)).to(torch.device('cuda'))

    # def cuda(self):
    #     super(AttentionLSTM, self).cuda()
    #     self.c_0 = self.c_0.cuda()
    #     return self
    
    def forward(self, support, query):
        # assert support.shape[-1] == query.shape[-1], "Support and query set have different embedding dimension!"
        # h, c 初始化是quert还是0还是随机?
        h = query
        c = self.c_0.expand_as(query)
        s_T = support.t()

        # self.num_layers 是超参数?
        for _ in range(self.num_layers):
            attentions = torch.mm(h, s_T)
            attentions = attentions.softmax(dim=1)
            readout    = torch.mm(attentions, support)
            x          = torch.cat((query, readout), 1)
            h, c       = self.lstm_cell(x, (h, c)) # 到底是哪两个拼接?
            h          = h + query

        h_norm = torch.norm(h, p=2, dim=1).unsqueeze(1).expand_as(h)
        h_normalized = h.div(h_norm + 0.00001)
        return h, h_normalized