import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as dist
import torch.optim as optim
import numpy as np


class Reinforce_agent(nn.Module):

    def __init__(self):
        super(Reinforce_agent, self).__init__()
        self.linear1 = nn.Linear(13, 20)
        self.linear2 = nn.Linear(20, 8)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        return x

    def act(self, state, items):
        out = self.forward(state)
        mean = out[0:4]
        std = torch.sqrt(out[4:8]**2)
        # std = out[4:8]
        distributions = torch.distributions.normal.Normal(mean, std) #STD should be positive
        proposal = distributions.sample()
        log_prob = distributions.log_prob(proposal)
        for i in range(len(proposal)):
            if proposal[i] > 1:
                proposal[i] = 1
            if proposal[i] < 0:
                proposal[i] = 0


        #print(log_prob)
        proposal = proposal.detach().numpy()
        termination = proposal[3]
        proposal = proposal[0:3]
        termination = 1 if termination > 0.5 else 0
        proposal = proposal*items
        proposal = proposal.astype(int)
        return termination, proposal, log_prob




