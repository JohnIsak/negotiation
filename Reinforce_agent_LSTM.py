import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as dist
import torch.optim as optim
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class Reinforce_agent(nn.Module):


    def __init__(self):
        super(Reinforce_agent, self).__init__()
        self.output_len = 13
        self.LSTM = nn.LSTM(13, 5, batch_first=True) # [BATCH SIZE, SEQ LENGTH, INPUT TENSOR]
        self.linear1 = nn.Linear(5, 100)
        self.linear2 = nn.Linear(100, self.output_len)
        self.h_n = None
        self.c_n = None

    def forward(self, x, mask):
        x, (h_n, c_n) = self.LSTM(x[mask], (self.h_n[:, mask], self.c_n[:, mask])) \
            if self.h_n is not None else self.LSTM(x, None) #x=[Batch, 1, x_dimensions]

        if self.h_n is None:
            self.h_n = h_n # [1, Batch Size, H_out]
            self.c_n = c_n # [1, Batch Size, H_out]
        else:
            self.h_n[:, mask] = h_n
            self.c_n[:, mask] = c_n

        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        return x

    def act(self, state, agent):
        out = self.forward(state, agent) # [Batch Size, seq_length, out_dim]
        out = torch.reshape(out, (-1, self.output_len))

        mean = out[:, 0:6]
        std = torch.abs(out[:, 6:12])
        cat_dist_val = out[:, 12]

        normal_distributions = torch.distributions.normal.Normal(mean, std) #STD should be positive
        sample = normal_distributions.sample()
        # print(sample)
        log_prob = normal_distributions.log_prob(sample)
        proposal = torch.sigmoid(sample[:, 0:3])
        utterance = sample[:, 3:6]
        log_prob[:, 0:3] *= proposal*(1-proposal)

        cat_dist_val = torch.sigmoid(cat_dist_val)
        cat_dist_val = torch.reshape(cat_dist_val, (-1, 1))
        cat_dist_val = torch.cat((cat_dist_val, 1-cat_dist_val), 1)
        cat_dist = torch.distributions.categorical.Categorical(cat_dist_val)
        cat_sample = cat_dist.sample()
        cat_log_prob = cat_dist.log_prob(cat_sample)
        cat_sample = torch.reshape(cat_sample, (-1, 1))
        cat_log_prob = torch.reshape(cat_log_prob, (-1, 1))

        log_prob = torch.cat((log_prob,cat_log_prob), 1)
        termination = cat_sample[:, 0].bool()
        return termination, proposal, log_prob, utterance




