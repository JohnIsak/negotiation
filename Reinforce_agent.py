import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as dist
import torch.optim as optim
import numpy as np


class Reinforce_agent(nn.Module):

    def __init__(self):
        super(Reinforce_agent, self).__init__()
        self.linear1 = nn.Linear(12, 20)
        self.linear2 = nn.Linear(20, 7)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x)
        out = self.linear2(x)
        return out

    def act(self, state, items):
        out = self.forward(state)
        termination = out[0] > 0
        proposal = torch.normal(out[1:4], out[4:7]**2) #STD should be positive
        proposal = dist.Normal(0, 1)
        proposal = proposal.detach().numpy()
        for i in range(len(proposal)):
            if proposal[i] > 1:
                proposal[i] = 1
            if proposal[i] < 0:
                proposal[i] = 0

        proposal = proposal*items
        proposal = proposal.astype(int)
        return termination, proposal




