import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=512, hidden_size=1024, bidirectional=True)

    def forward(self, X):
        input = X.transpose(0, 1)  # input : [n_step, batch_size, n_class]

        hidden_state = torch.zeros(1*2, len(X), 1024).to(torch.device('cuda'))   # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.zeros(1*2, len(X), 1024).to(torch.device('cuda'))     # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        return outputs.transpose(0, 1)