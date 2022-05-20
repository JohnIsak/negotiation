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
        self.LSTM = nn.LSTM(10, 100, batch_first=True) # [BATCH SIZE, SEQ LENGTH, INPUT TENSOR]
        self.linear1 = nn.Linear(100, 100)
        self.linear2 = nn.Linear(100, self.output_len)
        self.h_n = None
        self.c_n = None
        self.epsilon = 0.0000001


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

        mean = out[:, 0:6]
        std = torch.pow(out[:, 6:12], 2)
        std += self.epsilon #Can be 0 otherwise
        std[:, 3:6] = 0.001
        #print(std)
        if torch.rand(1) < 0.001:
           # print(std[:, 0:3], "standard deviations")
            print(mean[:, 3:6], "utterances")
        cat_dist_val = out[:, 12]

        if testing:
            proposal = torch.sigmoid(mean[:, 0:3])
            utterance = torch.tanh(mean[:, 3:6])
            termination = torch.ge(torch.sigmoid(cat_dist_val), 0.5)

            log_prob = None
            signal_loss = None

        else:
            normal_distributions = torch.distributions.normal.Normal(mean, std)  # STD should be positive
            # entropy = normal_distributions.entropy()
            # print(entropy)
            # print(entropy.shape)
            sample = normal_distributions.sample()
            log_prob = normal_distributions.log_prob(sample) #MÅ ENDRES HVIS ACTION SELECTION SKAL INKLUDERES FOR UTTPOL
            proposal = torch.sigmoid(sample[:, 0:3])
            utterance = torch.tanh(mean[:, 3:6])  # tanh

            cat_dist_val = torch.sigmoid(cat_dist_val)
            termination = torch.ge(torch.rand(len(cat_dist_val), device=device), cat_dist_val)
            cat_log_prob = termination.int() - cat_dist_val
            cat_log_prob = torch.reshape(cat_log_prob, (-1, 1))
            log_prob = torch.cat((log_prob, cat_log_prob), 1)


            signal_loss = self.positive_signalling_loss(utterance) if len(mean) > 2 else 0
        #Weighting of linguistic channel set to 0 when not using communication

        return termination, proposal, log_prob, utterance.clone().detach(), signal_loss

    def positive_signalling_loss(self, means):
        mask = torch.randint(2, (int(len(means)),), dtype=torch.bool)
        dist_1 = torch.pow(torch.abs(means[mask][:len(means[~mask])] - means[~mask][:len(means[mask])]) + self.epsilon, 2)
        dist_2 = torch.pow(2 - torch.abs((means[mask][:len(means[~mask])] - means[~mask][:len(means[mask])])), 2)
        dist = torch.cat((dist_1, dist_2), 1)
        dist = dist.resize(len(dist), 2, 3)
        delta = torch.min(dist[:], 1).values
        delta = torch.sqrt(torch.sum(delta, dim=1))
        delta = delta + self.epsilon
        loss = torch.mean(1/delta)
        loss = torch.nan_to_num(loss)
        return loss