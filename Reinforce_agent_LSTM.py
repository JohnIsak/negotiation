import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as dist
import torch.optim as optim
import numpy as np


class Reinforce_agent(nn.Module):
    hidden_a = None
    hidden_b = None

    def __init__(self):
        super(Reinforce_agent, self).__init__()
        self.LSTM = nn.LSTM(10, 100, batch_first=True)
        self.linear1 = nn.Linear(100, 100)
        self.linear2 = nn.Linear(100, 14)

    def forward(self, x, agent):
        if agent == 0:
            x, hidden = self.LSTM(x, self.hidden_a)
            self.hidden_a = hidden
        else:
            x, hidden = self.LSTM(x, self.hidden_b)
            self.hidden_b = hidden

        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        return x

    def act(self, state, agent):
        out = self.forward(state, agent)
        out = out[0, 0]
        #print(out)
        mean = out[0:7]
        std = torch.abs(out[7:14])
        #utterance = out[8:11]
        # std = out[4:8]
        distributions = torch.distributions.normal.Normal(mean, std) #STD should be positive
        proposal = distributions.sample()
        log_prob = distributions.log_prob(proposal)
        proposal[0:4] = F.sigmoid(proposal[0:4])

        #print(log_prob)
        # proposal = proposal.cpu().detach().numpy()
        utterance = proposal[4:7]
        termination = proposal[3]
        proposal = proposal[0:3]
        termination = 1 if termination > 0.5 else 0
        return termination, proposal, log_prob, utterance




