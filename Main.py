import numpy as np

import Negotiation
import Random_agent
import Reinforce_agent
import torch

agents = [Reinforce_agent.Reinforce_agent(), Reinforce_agent.Reinforce_agent()]
optimizers = [torch.optim.Adam(agents[0].parameters(), lr=0.0001), torch.optim.Adam(agents[1].parameters(), lr=0.0001)]
torch.autograd.set_detect_anomaly(True)

for i in range(10000):
    game = Negotiation.NegotiationGame()
    log_probs = [[], []]
    state_coded = game.state.generate_processed_state()
    agreement, proposal, log_prob = agents[0].act(state_coded, game.state.item_pool)
    log_probs[game.state.curr_player].append(log_prob)
    state, rewards = game.apply_action(proposal, None, agreement)
    while not state.is_terminal:
        state_coded = state.generate_processed_state()
        agreement, proposal, log_prob = agents[state.curr_player].act(state_coded, state.item_pool)
        log_probs[state.curr_player].append(log_prob)
        state, rewards = game.apply_action(proposal, None, agreement)
    for i in range(2):
        if log_probs[i]:
            loss = torch.cat(log_probs[i]).sum()
            #print(loss)
            loss = -loss * rewards[i]
            #print(loss)
            optimizers[i].zero_grad()
            loss.backward()
            optimizers[i].step()
    print(rewards)


#print(state.item_pool, state.hidden_utils, state.last_proposal, rewards, curr_player)
