import torch
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Reinforce_agent(nn.Module):
    def __init__(self, input_size, output_alphabet_size, output_length, signalling_loss, n_turns):
        super(Reinforce_agent, self).__init__()
        self.n_turns = n_turns
        self.output_length = output_length
        self.output_alphabet_size = output_alphabet_size
        self.encoder = nn.LSTM(input_size, 100, batch_first=True) # [BATCH SIZE, SEQ LENGTH, INPUT TENSOR]
        self.decoder = nn.LSTM(100+n_turns, 100, batch_first=True)
        self.fc = nn.Linear(100, output_alphabet_size)
        self.signalling_loss = signalling_loss
        self.target_entropy = torch.log(torch.tensor(self.output_alphabet_size))*0.1

    def forward(self, input, testing):
        x, hidden = self.encoder(input[:, :, :self.encoder.input_size]) # X: [BATCH SIZE, SEQ_LENGTH, HIDDEN_SIZE]
        x = x[:, -1:, :] # Only interested in final hidden state.
        turn_one_hot = F.one_hot((input[:, -1:, -1]).long(), num_classes=self.n_turns)
        encoded = torch.concat((x, turn_one_hot), dim=2)  # Appending current turn at the end of hidden state
        encoded = torch.repeat_interleave(encoded, self.output_length, 1) # Repeating input state output_length times

        y, hidden = self.decoder(encoded)
        y = self.fc(y)
        y = F.softmax(y, 2) # [Batch, seq_len, letter]
        if not testing:
            distributions = torch.distributions.Categorical(y)
            out = distributions.sample()
            #print(out.shape)
            log_probs = distributions.log_prob(out)
        else:
            out = torch.argmax(y, 2)
            #print(out.shape)
            return out, 0, 0

        signalling_loss = self.positive_signalling_loss(y) if self.signalling_loss else 0
        if self.signalling_loss == 0 and torch.rand(1) < 0.001:
            print(y, "y")
        return out, log_probs, signalling_loss

    def positive_signalling_loss(self, y):
        if len(y) < 2:
            return 0
        #self.target_entropy = 0.9999 * self.target_entropy
        target_entropy = self.target_entropy
        avg_policy = torch.mean(y, 0) #Denne ønsker man å ha så høy entropi som mulig med y = [Batch, seq_len, letter] [[0.33. 0.33 .33] [0.33
        #print(avg_policy)
        entropy_avg_policy = -torch.sum(avg_policy*torch.log(avg_policy))/self.output_length #Ønsker å maksimere
        conditioned_entropy = -torch.mean(y*torch.log(y))*self.output_alphabet_size
        loss = -entropy_avg_policy + ((conditioned_entropy-target_entropy)**2)
        # loss = -entropy_avg_policy + conditioned_entropy
        if torch.rand(1) < 0.001:
            print("target entropy", target_entropy)
            print(y, "y")
            print(avg_policy, "avg policy")
            print(entropy_avg_policy, "entropy_avg_policy")
            print(conditioned_entropy, "conditioned_entropy")
            print(target_entropy, "Target entropy")
            print(loss, "loss")
        return loss


    def positive_listening_loss(self, y):
        return NotImplementedError