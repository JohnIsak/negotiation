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
        self.output_len = 10
        self.LSTM = nn.LSTM(10, 100, batch_first=True) # [BATCH SIZE, SEQ LENGTH, INPUT TENSOR]
        self.linear1 = nn.Linear(100, 100)
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

    def act(self, state, mask, testing):
        out = self.forward(state, mask) # [Batch Size, seq_length, out_dim]
        out = torch.reshape(out, (-1, self.output_len))

        mean = out[:, 0:3]
        std = torch.pow(out[:, 3:6], 2)
        utterance = torch.tanh(out[:, 6:9])
        if torch.rand(1) < 0.001:
           # print(std[:, 0:3], "standard deviations")
            print(utterance, "utterances")
        cat_dist_val = out[:, 9]

        if testing:
            proposal = torch.sigmoid(mean[:, 0:3])

            termination = torch.ge(torch.sigmoid(cat_dist_val), 0.5)

            log_prob = None
            signal_loss = None

        else:
            normal_distributions = torch.distributions.normal.Normal(mean, std)  # STD should be positive
            # entropy = normal_distributions.entropy()
            # print(entropy)
            # print(entropy.shape)
            sample = normal_distributions.sample()
            log_prob = normal_distributions.log_prob(sample)
            proposal = torch.sigmoid(sample[:, 0:3])
            # utterance = torch.tanh(sample[:, 3:6])  # tanh

            cat_dist_val = torch.sigmoid(cat_dist_val)
            termination = torch.ge(torch.rand(len(cat_dist_val), device=device), cat_dist_val)
            cat_log_prob = termination.int() - cat_dist_val
            cat_log_prob = torch.reshape(cat_log_prob, (-1, 1))
            log_prob = torch.cat((log_prob, cat_log_prob), 1)


            signal_loss = self.positive_signalling_loss_naive(utterance) if len(mean) > 1 else 0


        #Weighting of linguistic channel set to 0 when not using communication

        return termination, proposal, log_prob, utterance.clone().detach(), signal_loss

    def positive_signalling_loss_naive(self, means):
        means_2 = means.resize(len(means), 1, 3).clone().detach()
        means_2 = means_2.repeat(1, len(means), 1)
        delta = torch.abs(means_2-means)
        mask = ~torch.diag(torch.ones(len(means), dtype=torch.bool))
        delta = delta[mask].reshape(len(means), len(means) - 1, 3)
        delta += 0.1
        loss = torch.mean(1/delta)
        return loss







