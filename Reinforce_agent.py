import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Reinforce_agent(nn.Module):
    def __init__(self):
        self.linear1 = nn.Linear(12, 20)
        self.linear2 = nn.Linear(20, 8)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x)
        out = self.linear2(x)
        return out

    def act(self, state):
        out = self.forward(state)
        termination = out[0] > out[1]
        proposal = torch.normal(out[2:5], out[5:8], dtype=torch.int)
        print(proposal, termination)



