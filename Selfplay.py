import numpy as np
import Negotiation_continuous
import Random_agent
import Reinforce_agent_LSTM
import torch
import matplotlib.pyplot as plt
import pandas as pd


def plot(rewards_saved):
    avg = pd.DataFrame(rewards_saved[:, 0])
    print(avg)
    avg = avg.iloc[:, 0].rolling(window=100).mean()
    avg_2 = pd.DataFrame((rewards_saved[:, 0]+ rewards_saved[:, 1])/2)
    avg_2 = avg_2.iloc[:, 0].rolling(window=100).mean()
    avg_3 = np.sort(rewards_saved, 1)
    avg_3 = pd.DataFrame(avg_3)
    avg_3_0 = avg_3.iloc[:, 0].rolling(window=100).mean()
    avg_3_1 = avg_3.iloc[:, 1].rolling(window=100).mean()

    plt.plot(avg)
    plt.title("Agent: " + str(0) + "\nFirst 100 000 iterations, Moving avg, window=100")
    plt.ylabel("Reward")
    plt.xlabel("Iteration")
    plt.show()
    plt.savefig("plot" + str(0))
    plt.plot(avg_2)
    plt.title("Both Agents mean \nFirst 100 000 iterations, Moving avg, window=100")
    plt.ylabel("Reward")
    plt.xlabel("Iteration")
    plt.show()
    plt.savefig("plot" + str(0))
    plt.plot(avg_3_0)
    plt.plot(avg_3_1)
    plt.title("Max-min Reward \nFirst 100 000 iterations, Moving avg, window=100")
    plt.ylabel("Reward")
    plt.xlabel("Iteration")
    plt.show()
    plt.savefig("plot" + str(0))



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

agents = [Reinforce_agent_LSTM.Reinforce_agent()]
agents[0] = agents[0].to(device=device)
optimizers = [torch.optim.Adam(agents[0].parameters(), lr=0.0001, weight_decay=0.0001)]
torch.autograd.set_detect_anomaly(True)
batch_size = 1
num_iterations = 500_000
print("asd1")

rewards_saved = torch.zeros((num_iterations, 2), device=device)
loss = torch.tensor(0.0, device=device)
print("asd1")


for i in range(num_iterations):
    # Play an episode
    game = Negotiation_continuous.NegotiationGame()
    state = game.state
    log_probs = [[], []]
    while not state.is_terminal:
        state_coded = state.state
        agreement, proposal, log_prob = agents[0].act(state_coded, state.curr_player)
        log_probs[state.curr_player].append(log_prob)
        state, rewards = game.apply_action(proposal, None, agreement)
    for agent in agents:
        agent.hidden_a = None
        agent.hidden_b = None
    print(log_probs)
    # Backprop
    if log_probs[0]:
        loss1 = torch.cat(log_probs[0]).sum()
        loss1 *= rewards[0]-0.4
        loss += loss1
    if log_probs[1]:
        loss2 = torch.cat(log_probs[1]).sum()
        loss2 *= rewards[1]-0.4
        loss += loss2
    loss = -loss
    optimizers[0].zero_grad()
    loss.backward()
    optimizers[0].step()
    loss = torch.tensor(0.0, device=device)
    print(rewards)
    rewards_saved[i] = rewards
torch.save(agents[0], "agent0_self")

rewards_saved = np.array(rewards_saved.cpu())
print(rewards_saved)
plot(rewards_saved)

#print(state.item_pool, state.hidden_utils, state.last_proposal, rewards, curr_player)
