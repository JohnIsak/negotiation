import numpy as np

import Negotiation
import Random_agent
import Reinforce_agent
import torch

a = Reinforce_agent.Reinforce_agent()
input = torch.arange(0, 1.2, 0.1)
a.act(input, np.array([4, 3, 5]))

game = Negotiation.NegotiationGame()
agents = [Reinforce_agent.Reinforce_agent(), Reinforce_agent.Reinforce_agent()]

state_coded = game.state.generate_processed_state()
agreement, proposal = agents[0].act(state_coded, game.state.item_pool)
state, rewards = game.apply_action(proposal, None, agreement)

while not state.is_terminal:
    state_coded = state.generate_processed_state()
    agreement, proposal = agents[state.curr_player].act(state_coded, state.item_pool)
    print(agreement)
    state, rewards = game.apply_action(proposal, None, agreement)

print(rewards)
#print(state.item_pool, state.hidden_utils, state.last_proposal, rewards, curr_player)
