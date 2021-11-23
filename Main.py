import numpy as np
import Negotiation
import Random_agent
import Reinforce_agent
import torch
import matplotlib.pyplot as plt
import pandas as pd

def plot(rewards_saved):
    for j in range(2):
        avg = pd.DataFrame(rewards_saved[:, j])
        print(avg)
        avg = avg.iloc[:, 0].rolling(window=100).mean()
        plt.plot(avg)
        plt.title("Agent: " + str(j) + "\nFirst 100 000 iterations, Moving avg, window=100")
        plt.ylabel("Reward")
        plt.xlabel("Iteration")
        plt.show()
        plt.savefig("plot" + str(j))

agents = [Reinforce_agent.Reinforce_agent(), Reinforce_agent.Reinforce_agent()]
agents[0] = agents[0].to(device=device)
optimizers = [torch.optim.Adam(agents[0].parameters(), lr=0.0001), torch.optim.Adam(agents[1].parameters(), lr=0.0001)]
torch.autograd.set_detect_anomaly(True)

rewards_saved = []

for i in range(100_000):
    # Play an episode
    game = Negotiation.NegotiationGame()
    state = game.state
    log_probs = [[], []]
    while not state.is_terminal:
        state_coded = state.generate_processed_state()
        agreement, proposal, log_prob = agents[state.curr_player].act(state_coded, state.item_pool)
        log_probs[state.curr_player].append(log_prob)
        state, rewards = game.apply_action(proposal, None, agreement)

    # Backprop
    for j in range(2):
        if log_probs[j]:
            loss = torch.cat(log_probs[j]).sum()
            loss = -loss * (rewards[j]-0.6075)
            optimizers[j].zero_grad()
            loss.backward()
            optimizers[j].step()

    rewards_saved.append(rewards)
torch.save(agents[0], "agent0")
torch.save(agents[1], "agent1")

rewards_saved = np.array(rewards_saved)
print(rewards_saved)
plot(rewards_saved)

#print(state.item_pool, state.hidden_utils, state.last_proposal, rewards, curr_player)
