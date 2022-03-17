import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as dist
import torch.optim as optim
import numpy as np


class Reinforce_agent(nn.Module):


    def __init__(self, n_players):
        super(Reinforce_agent, self).__init__()
        self.LSTM = nn.LSTM(13, 100, batch_first=True)
        self.linear1 = nn.Linear(100, 100)
        self.linear2 = nn.Linear(100, 13)
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

        #print(out)
        mean = out[0:6]
        std = torch.abs(out[6:12])
        cat_dist_val = out[12]

        normal_distributions = torch.distributions.normal.Normal(mean, std) #STD should be positive
        sample = normal_distributions.sample()
        log_prob = normal_distributions.log_prob(sample)
        proposal = torch.sigmoid(sample[0:3])
        utterance = sample[3:6]
        log_prob[0:3] *= proposal*(1-proposal)

        cat_dist_val = torch.sigmoid(cat_dist_val)
        cat_dist_val = torch.reshape(cat_dist_val, (-1, 1))
        cat_dist_val = torch.cat((cat_dist_val, 1-cat_dist_val), 1)
        cat_dist = torch.distributions.categorical.Categorical(cat_dist_val)
        cat_sample = cat_dist.sample()
        cat_log_prob = cat_dist.log_prob(cat_sample)

        log_prob = torch.cat((log_prob,cat_log_prob), 0)
        termination = cat_sample[0]
        return termination, proposal, log_prob, utterance




