import numpy as np
import Negotiation_continuous_more_players as Negotiation_continuous
import Random_agent
import Reinforce_agent_LSTM
import torch
import matplotlib.pyplot as plt
import pandas as pd


def reverse_order(rewards_saved, starting_player):
    for i in range(len(rewards_saved)):
        rewards_saved[i] = rewards_saved[i, starting_player[i]]
    return rewards_saved


def plot(rewards_saved, pos_reward):
    avg = pd.DataFrame(rewards_saved[:, 0])
    avg = avg.iloc[:, 0].rolling(window=100).mean()

    avg_2 = pd.DataFrame((rewards_saved[:, 0]+ rewards_saved[:, 1])/2)
    avg_2 = avg_2.iloc[:, 0].rolling(window=100).mean()

    max_min = np.sort(rewards_saved, 1)
    max_min = pd.DataFrame(max_min)

    max_min_plot = []
    for j in range(num_agents):
        max_min_plot.append(max_min.iloc[:, j].rolling(window=100).mean())

    rewards = pd.DataFrame(pos_reward)

    players = []
    for j in range(num_agents):
        players.append(rewards.iloc[:, j].rolling(window=100).mean())

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

    for val in max_min_plot:
        plt.plot(val)
    plt.title("Max-min Reward \nFirst 100 000 iterations, Moving avg, window=100")
    plt.ylabel("Reward")
    plt.xlabel("Iteration")
    plt.show()
    plt.savefig("plot" + str(0))

    for j, val in enumerate(players):
        label = "Player " + str(j+1)
        plt.plot(val, label=label)

    plt.title("Reward based on starting position, Moving avg, window=100")
    plt.ylabel("Reward")
    plt.xlabel("Iteration")
    plt.legend()
    plt.show()
    plt.savefig("plot" + str(0))


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_agents = 2
agents = [Reinforce_agent_LSTM.Reinforce_agent(num_agents)]
agents[0] = agents[0].to(device=device)
optimizers = [torch.optim.Adam(agents[0].parameters(), lr=0.0001, weight_decay=0.0001)]
#torch.autograd.set_detect_anomaly(True)
batch_size = 1
num_iterations = 200_000

starting_player = torch.zeros((num_iterations, num_agents), dtype=int)
rewards_saved = torch.zeros((num_iterations, num_agents), device=device)
max_reward_prosocial = torch.zeros(num_iterations)
loss = torch.tensor(0.0, device=device)
for i in range(num_iterations):
    # Play an episode
    game = Negotiation_continuous.NegotiationGame(num_agents)
    state = game.state
    for j in range(num_agents):
        starting_player[i][j] = (state.curr_player + j) % num_agents
    log_probs = [[],[]]
    while not state.is_terminal:
        state_coded = state.generate_processed_state()
        agreement, proposal, log_prob, utterance = agents[0].act(state_coded, state.curr_player)
        log_probs[state.curr_player].append(log_prob)
        state, rewards = game.apply_action(proposal, utterance, agreement)

    # Backprop
    for j in range(len(log_probs)):
        if log_probs[j]:
            loss_ = torch.cat(log_probs[j]).sum()
            loss_ *= rewards[j] - 1
            loss += loss_
    if (i+1) % batch_size == 0:
        loss = -loss
        optimizers[0].zero_grad()
        loss.backward()
        optimizers[0].step()
        loss = torch.tensor(0.0, device=device)
        for agent in agents:
            agent.hidden = np.empty(num_agents, dtype=tuple)
    if (i+1) % 100 == 0:
        print((i+1), rewards)
    rewards_saved[i] = rewards
    # rewards_saved[i] = rewards/(game.find_rewards(game.find_best_solution())-0.02)
torch.save(agents[0], "agent0_self")


rewards_saved = np.array(rewards_saved.cpu())
#print(rewards_saved)
pos_reward = reverse_order(rewards_saved.copy(), starting_player)
#print(rewards_saved)
plot(rewards_saved, pos_reward)

#print(state.item_pool, state.hidden_utils, state.last_proposal, rewards, curr_player)
