import torch
import torch.nn.functional as F
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class SequenceState:
    def __init__(self, batch_size, guess_alphabet_size, guess_seq_length, utt_albhabet_size, utt_seq_length, max_turns):
        self.batch_size = batch_size

        self.guess_alphabet_size = guess_alphabet_size
        self.guess_seq_length = guess_seq_length
        self.utt_alphabet_size = utt_albhabet_size
        self.utt_seq_length = utt_seq_length

        self.guess = torch.zeros((batch_size, guess_seq_length), device=device, dtype=torch.long)
        self.utt = torch.zeros((batch_size, utt_seq_length), device=device, dtype=torch.long)

        self.ans = torch.randint(guess_alphabet_size, (batch_size, guess_seq_length), device=device)

        self.still_alive = torch.ones(batch_size, dtype=torch.bool, device=device)

        self.turn = 0
        self.max_turns = max_turns
        self.curr_player = 0
        # 0 = Guesser, 1 = Mastermind

    def generate_processed_state(self):
        if self.curr_player == 0:
            state = torch.zeros((self.batch_size, self.utt_seq_length, self.utt_alphabet_size + self.max_turns + 1), dtype=torch.float, device=device)
            state[:, :, -self.max_turns-1:-1] = F.one_hot(torch.tensor(self.turn, device=device), num_classes=self.max_turns)
            state[:, :, -1] = self.turn/self.max_turns
        else:
            state = torch.zeros((self.batch_size, self.guess_seq_length, 2*self.guess_alphabet_size + self.max_turns + 1), dtype=torch.float,
                                device=device)
            state[:, :, :self.guess_alphabet_size] = F.one_hot(self.guess, num_classes=self.guess_alphabet_size)
            state[:, :, self.guess_alphabet_size:2*self.guess_alphabet_size] = F.one_hot(self.ans, num_classes=self.guess_alphabet_size)
            state[:, :, -self.max_turns-1:-1] = F.one_hot(torch.tensor(self.turn, device=device), num_classes=self.max_turns)
            state[:, :, -1] = self.turn/self.max_turns
        return state

class SequenceGame:
    def __init__(self, batch_size, guess_alphabet_size, guess_seq_length, utt_albhabet_size, utt_seq_length, max_turns):
        self.reward_sharing = 1
        self.batch_size = batch_size
        self.state = SequenceState(batch_size, guess_alphabet_size, guess_seq_length, utt_albhabet_size, utt_seq_length, max_turns)

    def apply_action(self, rewards, guess=None, utt=None):
        #print(self.state.turn, "Turn")
        #print(torch.sum(self.state.still_alive), "Number still alive")
        if self.state.curr_player == 0:
            finished = torch.ge(torch.sum(torch.eq(guess, self.state.ans[self.state.still_alive.clone()]).int(), 1), self.state.guess_seq_length)
            rewards_ = rewards[self.state.still_alive.clone()]
            rewards_[finished] += 1
            rewards[self.state.still_alive.clone()] = rewards_

            self.state.guess[self.state.still_alive.clone()] = guess
            self.state.still_alive[self.state.still_alive.clone()] = ~finished

            # THIS GIVES MAX TURNS GUESSES
            if self.state.turn == self.state.max_turns - 1:
                rewards = self.calculate_reward(rewards)
                self.state.still_alive[self.state.still_alive.clone()] = 0

            self.state.curr_player = (self.state.curr_player + 1) % 2

        else:
            self.state.utt[self.state.still_alive.clone()] = utt
            rewards[self.state.still_alive.clone()] -= 0.1
            self.state.turn += 1
            self.state.curr_player = (self.state.curr_player + 1) % 2
        return self.state, rewards

    def calculate_reward(self, rewards):
        rewards_ = rewards[self.state.still_alive.clone().detach()]
        rewards_[:, 0] += torch.sum(torch.eq(self.state.guess[self.state.still_alive.clone().detach()], self.state.ans[self.state.still_alive.clone().detach()]).int(), 1) \
                    * (1/self.state.guess_seq_length)
        rewards_[:, 1] += torch.sum(torch.eq(self.state.guess[self.state.still_alive.clone().detach()], self.state.ans[self.state.still_alive.clone().detach()]).int(), 1) \
                          * (1 / self.state.guess_seq_length)
        rewards[self.state.still_alive.clone()] = rewards_
        return rewards

