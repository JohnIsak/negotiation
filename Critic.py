import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.LSTM = nn.LSTM(13, 100, batch_first=True)
        self.linear1 = nn.Linear(100, 100)
        self.linear2 = nn.Linear(100, 1)
        self.hidden = None

    def forward(self, x):
        x, hidden = self.LSTM(x, self.hidden)
        self.hidden = hidden
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        x = x[0, 0]
        return x




