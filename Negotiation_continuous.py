import numpy as np
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
print(torch.version.cuda)

class NegotiationState:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.hidden_utils = torch.rand((batch_size, 2, 3), device=device)
        self.still_alive = torch.ones(batch_size, dtype=torch.bool, device=device)
        self.proposals = torch.zeros((batch_size, 3), device=device)
        self.utterances = torch.zeros((batch_size, 3), device=device)
        self.turn = 0
        self.curr_player = np.random.randint(0, 2)
        self.max_turns = 10
        self.remainder = torch.ones((batch_size, 3), device=device)

    def generate_processed_state(self):
        state = torch.zeros((self.batch_size, 10), dtype=torch.float, device=device)
        state[:, 0:3] = self.hidden_utils[torch.arange(0, self.batch_size), self.curr_player]
        # state[:, 3:6] = self.proposals
        state[:, 6:9] = self.utterances
        state[:, 9] = self.turn/self.max_turns
        state = torch.reshape(state, (-1, 1, 10))
        return state

class NegotiationGame:

    def __init__(self, batch_size):
        self.reward_sharing = 1
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
        curr_utils = self.state.hidden_utils[(torch.arange(0, self.batch_size), self.state.curr_player)]
        other_utils = self.state.hidden_utils[(torch.arange(0, self.batch_size),(self.state.curr_player+1) % 2)]
        best_proposal = torch.ge(curr_utils, other_utils).int()
        return best_proposal

    def find_best_solution_different(self):
        curr_utils = self.state.hidden_utils[(torch.arange(0, self.batch_size), self.state.curr_player)]
        other_utils = self.state.hidden_utils[(torch.arange(0, self.batch_size), (self.state.curr_player + 1) % 2)]
        best_proposal_curr = torch.ge(curr_utils, other_utils*self.reward_sharing).int()
        best_proposal_other = torch.ge(curr_utils*self.reward_sharing, other_utils).int()
        return best_proposal_curr, best_proposal_other

    def apply_action(self, proposals, utterances, agreement, rewards):

        if self.state.turn < 1:
            agreement = torch.zeros(agreement.shape, dtype=torch.bool, device=device)

        # Max Turns reached
        if self.state.turn == self.state.max_turns:
            # agreement = torch.ones(agreement.shape, dtype=torch.bool, device=device)
            rewards[self.state.still_alive] = -1
            self.state.still_alive[self.state.still_alive] = 0
            return self.state, rewards

        self.state.turn += 1
        self.state.curr_player = (self.state.curr_player + 1) % 2

        if self.state.turn > 1:
            rewards = self.find_rewards_prosocial(agreement, rewards)

        self.state.proposals[self.state.still_alive] = proposals
        self.state.utterances[self.state.still_alive] = utterances
        self.state.still_alive[self.state.still_alive.clone()] = ~agreement
        return self.state, rewards

    def find_rewards(self, agreement, rewards, proposal=None):
        curr_player = self.state.curr_player
        other_player = (self.state.curr_player + 1) % 2

        rewards_ = rewards[self.state.still_alive]
        proposals = self.state.proposals[self.state.still_alive] if proposal is None else proposal

        hidden_utils = self.state.hidden_utils[self.state.still_alive]
        ones = torch.ones((torch.sum(agreement), 3), device=device)

        rewards_[agreement, curr_player] = torch.sum(proposals[agreement] * hidden_utils[agreement, curr_player],  dim=1)
        rewards_[agreement, other_player] = torch.sum((ones - proposals[agreement]) * hidden_utils[agreement, other_player], dim=1)

        a = rewards_[agreement, curr_player].clone().detach()
        rewards_[agreement, curr_player] += self.reward_sharing*rewards_[agreement, other_player]
        rewards_[agreement, other_player] += self.reward_sharing*a
        return rewards_

    def find_rewards_prosocial(self, agreement, rewards):
        rewards_players = self.find_rewards(agreement, rewards)
        # best_proposals = self.find_best_solution()
        best_proposal_curr, best_proposal_other = self.find_best_solution_different()
        rewards_max_curr = self.find_rewards(agreement, rewards, best_proposal_curr[self.state.still_alive])
        rewards_max_other = self.find_rewards(agreement, rewards, best_proposal_other[self.state.still_alive])
        # rewards_max = self.find_rewards(agreement, rewards, best_proposals[self.state.still_alive])
        # rewards_ = rewards_players[agreement]/rewards_max[agreement]
        rewards_ = rewards_players[agreement]
        rewards_[:, self.state.curr_player] /= rewards_max_curr[agreement, self.state.curr_player]
        rewards_[:, (self.state.curr_player+1)%2] /= rewards_max_other[agreement, (self.state.curr_player+1)%2]
        #print(rewards_)
        rewards_ = 1/(np.e**(-5*(rewards_-1)))

        a = rewards[self.state.still_alive]
        a[agreement] = rewards_
        rewards[self.state.still_alive] = a
        return rewards