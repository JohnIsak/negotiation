import numpy as np
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
print(torch.version.cuda)

class NegotiationState:
    def __init__(self, num_players):
        self.num_players = num_players

        utils_generated = False
        while not utils_generated:
            self.hidden_utils = torch.randint(0, 11, (self.num_players, 3), device=device)
            utils_generated = True
            for i in range(num_players):
                if sum(self.hidden_utils[i]) == 0:
                    utils_generated = False

        self.is_terminal = False
        self.last_proposal = None
        self.last_utterance = None
        self.next_last_proposal = None
        self.turn = 0
        self.curr_player = np.random.randint(0, 3)
        self.max_turns = 20
        self.agreement_counter = 0

    def generate_processed_state(self):
        state = torch.zeros(10, dtype=torch.float, device=device)
        state[0:3] = self.hidden_utils[self.curr_player]/10
        # print(type(self.last_proposal))
        # if self.last_proposal is not None:
        #    state[3:6] = self.last_proposal
        state[6:9] = self.last_utterance if self.last_utterance is not None else state[6:9]
        state[9] = self.turn/20
        # state = torch.tensor(state, dtype=torch.float, device=device)
        state = torch.reshape(state, (1, 1, -1))
        return state


class NegotiationGame:

    def __init__(self):
        self.state = NegotiationState()

    def _find_max_utility(self):
        max_utils = [self.state.hidden_utils[0, i] if self.state.hidden_utils[0, i] > self.state.hidden_utils[1, i]
                     else self.state.hidden_utils[1, i] for i in range(len(self.state.hidden_utils[0]))]
        return np.sum(max_utils)

    def _find_max_self_interest(self):
        max_self_interest = torch.zeros(self.state.num_players, device=device)
        for i in range(max_self_interest):
            max_self_interest[i] = torch.sum(self.state.hidden_utils[i])
        return max_self_interest

    def apply_action(self, proposal, utterance, agreement):
        self.state.turn += 1
        self.state.curr_player = (self.state.curr_player + 1) % self.state.num_players
        if agreement:
            self.state.is_terminal = True
            if self.state.last_proposal is None:
                return self.state, torch.ones(2, device=device)* -1
            rewards = self.find_rewards()
            return self.state, rewards
        if self.state.turn == self.state.max_turns:
            self.state.is_terminal = True
            return self.state, torch.ones(2, device=device) * -1

        self.state.next_last_proposal = self.state.last_proposal
        self.state.last_proposal = proposal
        self.state.last_utterance = utterance
        return self.state, torch.zeros(2, device=device)

    def find_rewards(self, proposal=None):
        other_player = (self.state.curr_player + 1) % 2
        last_proposal = self.state.last_proposal if proposal is None else proposal
        rewards = torch.zeros(2, device=device)
        maximum_rewards = self._find_max_self_interest()

        # Rewards til spilleren som kom med proposalet
        # Det er spilleren som er current player.
        # curr-player1-> proposal -> curr-player2-> acceptance -> curr-player1

        rewards[self.state.curr_player] = torch.sum(last_proposal * self.state.hidden_utils[self.state.curr_player]) \
            / maximum_rewards[self.state.curr_player]
        rewards[other_player] = torch.sum((torch.ones(3, device=device) - last_proposal)
                                           * self.state.hidden_utils[other_player]) / maximum_rewards[other_player]
        a = rewards[self.state.curr_player].clone().detach()
        rewards[self.state.curr_player] += 1*rewards[other_player]
        rewards[other_player] += 1*a
        rewards -= 0.01*self.state.turn
        return rewards
