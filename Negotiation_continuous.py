import numpy as np
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class NegotiationState:
    # State (A,B) A = Current Player, B State for that Player
    # State 0:3 = Hidden Utils
    # State 3:6 = Last Proposal
    def __init__(self):
        self.state = torch.zeros((2, 6), dtype=torch.float, device=device)
        self.state[:, 0:3] = torch.randint(0, 11, (2, 3), device=device)
        while sum(self.state[0]) == 0 or sum(self.state[1]) == 0:
            self.state[:, 0:3] = torch.randint(0, 11, (2, 3), device=device)
        self.state[:, 0:3] = self.state[:, 0:3]/10
        self.is_terminal = False
        # self.last_proposal = None
        # self.last_utterance = None
        self.turn = 0
        self.curr_player = np.random.randint(0, 2)
        self.max_turns = 20


    def generate_processed_state(self):
        state = torch.zeros(6, dtype=torch.float, device=device)
        state[0:3] = self.hidden_utils[self.curr_player]/10
        # print(type(self.last_proposal))
        if self.last_proposal is not None:
            state[3:6] = self.last_proposal
        # state[6:9] = self.next_last_proposal/5 if self.next_last_proposal is not None else state[9:12]
        # state[6] = self.turn/20
        # state = torch.tensor(state, dtype=torch.float, device=device)
        state = torch.reshape(state, (1, 1, -1))
        return state


class NegotiationGame:

    def __init__(self):
        self.state = NegotiationState()

    def _find_max_utility(self):
        max_utils = [self.state.state[0, i] if self.state.state[0, i] > self.state.state[1, i]
                     else self.state.state[1, i] for i in range(3)]
        return np.sum(max_utils)

    def _find_max_self_interest(self):
        max_self_interest = torch.zeros(2, device=device)
        max_self_interest[0] = torch.sum(self.state.state[0, 0:3])
        max_self_interest[1] = torch.sum(self.state.state[1, 0:3])
        return max_self_interest

    def find_best_solution(self):
        best_proposal = np.zeros(3)
        for i in range(len(best_proposal)):
            if self.state.state[self.state.curr_player][i] \
                    > self.state.state[(self.state.curr_player+1) % 2][i]:
                best_proposal[i] = 1
        return best_proposal

    def apply_action(self, proposal, utterance, agreement):
        self.state.turn += 1
        self.state.curr_player = (self.state.curr_player + 1) % 2

        if agreement:
            self.state.is_terminal = True
            if sum(self.state.state[0, 3:6] == 0):
                return self.state, torch.ones(2, device=device)* -1
            rewards = self.find_rewards()
            return self.state, rewards
        if self.state.turn == self.state.max_turns:
            self.state.is_terminal = True
            return self.state, torch.ones(2, device=device) * -1

        self.state.state[self.state.curr_player, 3:6] = proposal
#         self.state.last_utterance = utterance
        return self.state, torch.zeros(2, device=device)

    def find_rewards(self):
        other_player = (self.state.curr_player + 1) % 2
        last_proposal = self.state.state[self.state.curr_player, 3:6]
        rewards = torch.zeros(2, device=device)
        maximum_rewards = self._find_max_self_interest()

        # Rewards til spilleren som kom med proposalet
        # Det er spilleren som er current player.
        # curr-player1-> proposal -> curr-player2-> acceptance -> curr-player1
        rewards[self.state.curr_player] = torch.sum(last_proposal * self.state.state[self.state.curr_player][0:3]) \
            / maximum_rewards[self.state.curr_player]
        rewards[other_player] = torch.sum((torch.ones(3, device=device) - last_proposal)
                                           * self.state.state[other_player][0:3]) / maximum_rewards[other_player]
        a = rewards[self.state.curr_player]
        rewards[self.state.curr_player] += 0.0*rewards[other_player]
        rewards[other_player] += 0.0*a
        rewards -= 0.05*self.state.turn
        return rewards
