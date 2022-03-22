import torch.nn.functional as F
import torch.nn as nn


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.LSTM = nn.LSTM(10, 100, batch_first=True)
        self.linear1 = nn.Linear(100, 100)
        self.linear2 = nn.Linear(100, 1)
        self.h_n = None
        self.c_n = None

    def forward(self, x, mask):
        x, (h_n, c_n) = self.LSTM(x[mask], (self.h_n[:, mask], self.c_n[:, mask])) \
            if self.h_n is not None else self.LSTM(x, None)  # x=[Batch, 1, x_dimensions]

        if self.h_n is None:
            self.h_n = h_n  # [1, Batch Size, H_out]
            self.c_n = c_n  # [1, Batch Size, H_out]
        else:
            self.h_n[:, mask] = h_n
            self.c_n[:, mask] = c_n

        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x) # [1, Batch Size, H_out]
        x = x.reshape((-1, 1))
        return x




