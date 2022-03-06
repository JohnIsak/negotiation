import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class Critic(nn.Module):

    def __init__(self, n_players):
        super(Critic, self).__init__()
        self.LSTM = nn.LSTM(13, 100, batch_first=True)
        self.linear1 = nn.Linear(100, 100)
        self.linear2 = nn.Linear(100, 1)
        self.hidden = np.empty(n_players, dtype=tuple)

    def forward(self, x, agent):
        x, hidden = self.LSTM(x, self.hidden[agent])
        self.hidden[agent] = hidden
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        return x

    def act(self, state, agent):
        out = self.forward(state, agent)
        out = out[0, 0]
        return out




