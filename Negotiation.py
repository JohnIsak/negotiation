import numpy as np
import torch


class NegotiationState:

    def __init__(self):
        self.item_pool = np.random.randint(0, 6, 3)
        while sum(self.item_pool) == 0:
            self.item_pool = np.random.randint(0, 6, 3)

        self.hidden_utils = np.random.randint(0, 11, (2, 3))
        while sum(self.hidden_utils[0] * self.item_pool) == 0 or sum(self.hidden_utils[1] * self.item_pool) == 0:
            self.hidden_utils = np.random.randint(0, 11, (2, 3))

        self.is_terminal = False
        self.last_proposal = None
        self.last_utterance = None
        self.next_last_proposal = None
        self.turn = 0
        self.curr_player = np.random.randint(0, 2)
        self.max_turns = np.random.randint(5, 11)

    def generate_processed_state(self):
        state = np.zeros(13)
        state[0:3] = self.item_pool/5
        state[3:6] = self.hidden_utils[self.curr_player]/10
        state[6:9] = self.last_proposal/5 if self.last_proposal is not None else state[6:9]
        state[9:12] = self.next_last_proposal/5 if self.next_last_proposal is not None else state[9:12]
        state[12] = self.turn/10
        state = torch.tensor(state, dtype=torch.float)
        return state


class NegotiationGame:

    def __init__(self):
        self.state = NegotiationState()

    def _find_max_utility(self):
        max_utils = [self.state.hidden_utils[0, i] if self.state.hidden_utils[0, i] > self.state.hidden_utils[1, i]
                     else self.state.hidden_utils[1, i] for i in range(len(self.state.hidden_utils[0]))]
        return np.sum(max_utils * self.state.item_pool)

    def _find_max_self_interest(self):
        max_self_interest = np.zeros(2)
        max_self_interest[0] = np.sum(self.state.hidden_utils[0] * self.state.item_pool)
        max_self_interest[1] = np.sum(self.state.hidden_utils[1] * self.state.item_pool)
        return max_self_interest

    def find_best_solution(self):
        best_proposal = np.zeros(3)
        for i in range(len(best_proposal)):
            if self.state.hidden_utils[self.state.curr_player][i] \
                    > self.state.hidden_utils[(self.state.curr_player+1) % 2][i]:
                best_proposal[i] = self.state.item_pool[i]
        return best_proposal

    def is_legal_proposal(self, proposal):
        return np.sum(np.less_equal(proposal, self.state.item_pool)) == len(proposal)

    def apply_action(self, proposal, utterance, agreement):
        self.state.turn += 1
        self.state.curr_player = (self.state.curr_player + 1) % 2

        if not self.is_legal_proposal(proposal):
            self.state.is_terminal = True
            return self.state, np.repeat(-1, 2)
        if agreement:
            self.state.is_terminal = True
            if self.state.last_proposal is None:
                return self.state, np.repeat(-1, 2)
            rewards = self.find_rewards()
            return self.state, rewards
        if self.state.turn == self.state.max_turns:
            self.state.is_terminal = True
            return self.state, np.repeat(-1, 2)

        self.state.next_last_proposal = self.state.last_proposal
        self.state.last_proposal = proposal
        self.state.last_utterance = utterance
        return self.state, np.repeat(0, 2)

    def find_rewards(self):
        other_player = (self.state.curr_player + 1) % 2
        rewards = np.zeros(2)
        maximum_rewards = self._find_max_self_interest()

        # Rewards til spilleren som kom med proposalet
        # Det er spilleren som er current player.
        # curr-player1-> proposal -> curr-player2-> acceptance -> curr-player1
        rewards[self.state.curr_player] = np.sum(self.state.last_proposal * self.state.hidden_utils[self.state.curr_player]) \
            / maximum_rewards[self.state.curr_player]
        rewards[other_player] = np.sum((self.state.item_pool - self.state.last_proposal)
                                           * self.state.hidden_utils[other_player]) / maximum_rewards[other_player]
        return rewards
