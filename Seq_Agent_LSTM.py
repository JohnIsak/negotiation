import torch
import torch.nn.functional as F
import torch.nn as nn


class Reinforce_agent(nn.Module):
    def __init__(self, input_size, output_alphabet_size, output_length):
        super(Reinforce_agent, self).__init__()
        self.output_length = output_length
        self.encoder = nn.LSTM(input_size, 10, batch_first=True) # [BATCH SIZE, SEQ LENGTH, INPUT TENSOR]
        self.decoder = nn.LSTM(11, output_alphabet_size, batch_first=True)

    def forward(self, input):
        x, hidden = self.encoder(input[:, :, :self.encoder.input_size], None) #X: [BATCH SIZE, SEQ_LENGTH, HIDDEN_SIZE]
        x = x[:, -1:, :]
        encoded = torch.concat((x, input[:, -1:, -1:]), dim=2)  #Appending current turn at the end
        encoded = torch.repeat_interleave(encoded, self.output_length, 1)
        y, hidden = self.decoder(encoded)
        y = F.softmax(y, 2)
        distributions = torch.distributions.Categorical(y)
        out = distributions.sample()
        log_probs = distributions.log_prob(out)
        return out, log_probs



