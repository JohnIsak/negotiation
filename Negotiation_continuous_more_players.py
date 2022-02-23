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
        self.proposals = []
        self.last_utterance = None
        self.turn = 0
        self.curr_player = np.random.randint(0, self.num_players)
        self.max_turns = 20
        self.agreement_counter = 0
        self.remainder = torch.ones(3, device=device)


    def generate_processed_state(self):
        state = torch.zeros(13, dtype=torch.float, device=device)
        state[0:3] = self.hidden_utils[self.curr_player]/10
        # print(type(self.last_proposal))
        # if len(self.proposals) > 0:
        #    state[3:6] = self.proposals[-1]
        state[6:9] = self.last_utterance if self.last_utterance is not None else state[6:9]
        # state[9:12] = self.remainder
        state[12] = self.turn/20
        state = torch.reshape(state, (1, 1, -1))
        return state


class NegotiationGame:

    def __init__(self, num_players):
        self.state = NegotiationState(num_players)

    def _find_max_utility(self):
        max_utils = [self.state.hidden_utils[0, i] if self.state.hidden_utils[0, i] > self.state.hidden_utils[1, i]
                     else self.state.hidden_utils[1, i] for i in range(len(self.state.hidden_utils[0]))]
        return np.sum(max_utils)

    def _find_max_self_interest(self):
        max_self_interest = torch.zeros(self.state.num_players, device=device)
        for i in range(len(max_self_interest)):
            max_self_interest[i] = torch.sum(self.state.hidden_utils[i])
        return max_self_interest

    def apply_action(self, proposal, utterance, agreement):

        if self.state.turn == self.state.max_turns:
            self.state.is_terminal = True
            return self.state, torch.ones(self.state.num_players, device=device) * -1

        self.state.turn += 1
        self.state.curr_player = (self.state.curr_player + 1) % self.state.num_players
        if agreement and len(self.state.proposals) > 0:
            self.state.agreement_counter += 1
            if self.state.agreement_counter == self.state.num_players-1:
                self.state.is_terminal = True
                rewards = self.find_rewards()
                return self.state, rewards
            self.state.remainder = self.state.remainder-proposal
            self.state.proposals.append(proposal)
            self.state.last_utterance = utterance
            return self.state, torch.zeros(self.state.num_players, device=device)


        self.state.agreement_counter = 0
        self.state.remainder = torch.ones(3, device=device)
        self.state.proposals = [proposal]
        self.state.last_utterance = utterance
        return self.state, torch.zeros(self.state.num_players, device=device)

    def find_rewards(self, proposal=None):
        players = np.zeros(self.state.num_players, dtype=int)
        player = self.state.curr_player
        for i in range(self.state.num_players):
            players[i] = (player+i) % self.state.num_players

        proposals = self.state.proposals
        rewards = torch.zeros(self.state.num_players, device=device)
        maximum_rewards = self._find_max_self_interest()

        remainder = torch.ones(3, device=device)

        # Spiller 1 kommer med proposal
        for i in range(len(players)-1):
            rewards[players[i]] = torch.sum(remainder * proposals[i] * self.state.hidden_utils[players[i]]) \
                                  / maximum_rewards[players[i]]
            remainder = remainder-remainder*proposals[i]

        #Final player gets remaining
        rewards[players[-1]] = torch.sum(remainder * self.state.hidden_utils[players[-1]]) \
                                  / maximum_rewards[players[-1]]

        a = rewards.clone().detach()
        for j in range(self.state.num_players):
            mask = np.ones(self.state.num_players, dtype=bool)
            mask[j] = False
            rewards[j] += torch.sum(1*a[mask])
        #rewards[self.state.curr_player] += 1*rewards[other_player]
        #rewards[other_player] += 1*a
        rewards -= 0.05*self.state.turn
        return rewards
