import Negotiation
import Random_agent
import Reinforce_agent
import torch

a = Reinforce_agent
input = torch.arange(1, 14)
print(input)

game = Negotiation.NegotiationGame()
agents = [Random_agent.RandomAgent(game.state.item_pool), Random_agent.RandomAgent(game.state.item_pool)]

proposal, agreement = agents[0].act()
state, rewards, curr_player = game.apply_action(proposal, None, agreement)

while not state.is_terminal:
    proposal, agreement = agents[curr_player].act()
    state, rewards, curr_player = game.apply_action(proposal, None, agreement)

#print(state.item_pool, state.hidden_utils, state.last_proposal, rewards, curr_player)
