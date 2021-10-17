import numpy as np
import torch


class NegotiationState:

    def __init__(self):
        self.item_pool = np.random.randint(0, 6, 3)
        self.hidden_utils = np.random.randint(0, 11, (2, 3))
        self.is_terminal = False
        self.last_proposal = None
        self.last_utterance = None
        self.next_last_proposal = None
        self.turn = 0
        self.curr_player = 0
        self.max_turns = np.random.randint(5, 11)

    def generate_processed_state(self):
        state = np.zeros(12)
        state[0:3] = self.item_pool/5
        state[3:6] = self.hidden_utils[self.curr_player]/10
        state[6:9] = self.last_proposal/5 if self.last_proposal is not None else state[6:9]
        state[9:12] = self.next_last_proposal/5 if self.next_last_proposal is not None else state[9:12]
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
        self.state.next_last_proposal = self.state.last_proposal
        self.state.last_proposal = proposal
        self.state.last_utterance = utterance
        return self.state, np.repeat(0, 2)

    def find_rewards(self):
        other_player = (self.state.curr_player + 1) % 2

        rewards = np.zeros(2)
        maximum_rewards = self._find_max_self_interest()
        rewards[other_player] = np.sum(self.state.last_proposal * self.state.hidden_utils[other_player]) \
            / maximum_rewards[other_player]
        rewards[self.state.curr_player] = np.sum((self.state.item_pool - self.state.last_proposal)
                                           * self.state.hidden_utils[self.state.curr_player]) / \
            maximum_rewards[self.state.curr_player]
        return rewards
