import numpy as np
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
print(torch.version.cuda)

class NegotiationState:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        # TODO Make sure [0,0,0] is not generated
        self.hidden_utils = torch.rand((batch_size, 2, 3), device=device)
        self.still_alive = torch.ones(batch_size, dtype=torch.bool, device=device)

        self.proposals = torch.zeros((batch_size, 3), device=device)
        self.utterances = torch.zeros((batch_size, 3), device=device)
        self.turn = 0
        self.curr_player = torch.randint(0, 2, (batch_size,))
        self.max_turns = 10
        self.remainder = torch.ones((batch_size, 3), device=device)

    def generate_processed_state(self):
        state = torch.zeros((self.batch_size, 10), dtype=torch.float, device=device)
        state[:, 0:3] = self.hidden_utils[torch.arange(0, self.batch_size), self.curr_player]
        # TODO Legg til batching for proposals også.
        # print(type(self.last_proposal))
        #if len(self.proposals) > 0:
        #    state[3:6] = self.proposals[-1]
        state[:, 6:9] = self.utterances
        state[:, 9] = self.turn/self.max_turns
        state = torch.reshape(state, (-1, 1, 10))
        return state, self.still_alive


class NegotiationGame:

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.state = NegotiationState(batch_size)

    def _find_max_utility(self):
        max_utils = [self.state.hidden_utils[0, i] if self.state.hidden_utils[0, i] > self.state.hidden_utils[1, i]
                     else self.state.hidden_utils[1, i] for i in range(len(self.state.hidden_utils[0]))]
        return np.sum(max_utils)

    def _find_max_self_interest(self):
        max_self_interest = torch.zeros(2, device=device)
        max_self_interest[0] = torch.sum(self.state.hidden_utils[0])
        max_self_interest[1] = torch.sum(self.state.hidden_utils[1])
        return max_self_interest

    def find_best_solution(self):
        best_proposal = torch.zeros(3, device=device)
        for i in range(len(best_proposal)):
            if self.state.hidden_utils[self.state.curr_player][i] \
                    > self.state.hidden_utils[(self.state.curr_player+1) % 2][i]:
                best_proposal[i] = 1
        return best_proposal

    def apply_action(self, proposals, utterances, agreement, rewards):

        if self.state.turn < 1:
            agreement = torch.zeros(agreement.shape, dtype=torch.bool, device=device)

        # Max Turns reached
        if self.state.turn == self.state.max_turns:
            rewards[self.state.still_alive] = -1
            self.state.still_alive[self.state.still_alive] = 0
            return self.state, rewards, self.state.still_alive

        self.state.turn += 1
        self.state.curr_player = (self.state.curr_player + 1) % 2

        if self.state.turn > 1:
            rewards = self.find_rewards(agreement, rewards)

        self.state.proposals[self.state.still_alive] = proposals
        self.state.utterances[self.state.still_alive] = utterances
        self.state.still_alive[self.state.still_alive] = ~agreement
        return self.state, rewards, self.state.still_alive

    def find_rewards(self, agreement, rewards):
        curr_player = self.state.curr_player[self.state.still_alive][agreement]
        other_player = ((self.state.curr_player + 1) % 2)[self.state.still_alive][agreement]
        rewards_ = rewards[self.state.still_alive]
        proposals = self.state.proposals[self.state.still_alive]
        print(rewards_, "rewards_")
        print(agreement, "agreement")
        hidden_utils = self.state.hidden_utils[self.state.still_alive]
        ones = torch.ones((torch.sum(agreement), 3), device=device)

        rewards_[agreement, curr_player] = torch.sum(proposals[agreement] * hidden_utils[agreement, curr_player],  dim=1)
        rewards_[agreement, other_player] = torch.sum((ones - proposals[agreement]) * hidden_utils[agreement, other_player], dim=1)

        a = rewards_[agreement, curr_player].clone().detach()
        rewards_[agreement, curr_player] += 1*rewards_[agreement, other_player]
        rewards_[agreement, other_player] += 1*a
        rewards[self.state.still_alive] = rewards_
        return rewards
