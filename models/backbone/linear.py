import torch.nn as nn

# Basic linear: y = x A^\top + b
class MyLinear(nn.Module):
    def __init__(self, feature_dim=3 * 84 * 84, hid_dim=64):
        super(MyLinear, self).__init__()
        self.encoder = nn.Linear(feature_dim, hid_dim)

    def forward(self, x):
        all_but_last_three_dims = x.size()[:-3]
        x = x.view(*all_but_last_three_dims, -1)
        x = self.encoder(x)
        return x
