import numpy as np
import Negotiation_continuous_more_players as Negotiation_continuous
import Reinforce_agent_LSTM
import Critic
import torch
import matplotlib.pyplot as plt
import pandas as pd


def reverse_order(rewards_saved, starting_player):
    for i in range(len(rewards_saved)):
        rewards_saved[i] = rewards_saved[i, starting_player[i]]
    return rewards_saved


def plot(rewards_saved, pos_reward):
    avg = pd.DataFrame(rewards_saved[:, 0])
    avg = avg.iloc[:, 0].rolling(window=500).mean()

    avg_2 = pd.DataFrame((rewards_saved[:, 0]+ rewards_saved[:, 1])/2)
    avg_2 = avg_2.iloc[:, 0].rolling(window=500).mean()

    max_min = np.sort(rewards_saved, 1)
    max_min = pd.DataFrame(max_min)

    max_min_plot = []
    for j in range(num_agents):
        max_min_plot.append(max_min.iloc[:, j].rolling(window=500).mean())

    rewards = pd.DataFrame(pos_reward)

    players = []
    for j in range(num_agents):
        players.append(rewards.iloc[:, j].rolling(window=500).mean())

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
agents = [Reinforce_agent_LSTM.Reinforce_agent(num_agents), Reinforce_agent_LSTM.Reinforce_agent(num_agents)]
critic = Critic.Critic(13)
critic = critic.to(device=device)
q_critic = Critic.Critic(13+7)
q_critic = q_critic.to(device=device)

agents[0] = agents[0].to(device=device)
agents[1] = agents[1].to(device=device)
optimizers = [torch.optim.Adam(agents[0].parameters(), lr=0.0001),
              torch.optim.Adam(agents[1].parameters(), lr=0.0001)]

optimizer_critic = torch.optim.Adam(critic.parameters(), lr=0.0001)
optimizer_q_critic = torch.optim.Adam(q_critic.parameters(), lr=0.0001)
torch.autograd.set_detect_anomaly(True)
batch_size = 1

num_iterations = 200_000

starting_player = torch.zeros((num_iterations, num_agents), dtype=int)
rewards_saved = torch.zeros((num_iterations, num_agents), device=device)
max_reward_prosocial = torch.zeros(num_iterations)
losses = []
for i in range(num_iterations):
    # Play an episode
    game = Negotiation_continuous.NegotiationGame(num_agents)
    state = game.state
    for j in range(num_agents):
        starting_player[i][j] = (state.curr_player + j) % num_agents
    log_probs = [[], []]

    q_values = torch.empty(2, device=device)
    v = torch.empty(2, device=device)

    while not state.is_terminal:
        state_coded = state.generate_processed_state()


        agreement, proposal, log_prob, utterance = agents[state.curr_player].act(state_coded, state.curr_player)

        v[state.curr_player] = critic(state_coded)

        s_a = torch.cat((state_coded[0][0], torch.tensor([agreement], device=device), proposal, utterance), 0)
        s_a = torch.reshape(s_a, (1, 1, -1))
        q_values[state.curr_player] = q_critic(s_a)

        #print(agreement, proposal, utterance)
        log_probs[state.curr_player].append(log_prob)
        state, rewards = game.apply_action(proposal, utterance, agreement)

    rewards = rewards / game.find_rewards([game.best_collective_proposal()])
    rewards -= 0.01 * state.turn

    with torch.no_grad():
        delta = q_values - v

    loss_critic = torch.sum((rewards - v)**2)
    loss_q_critic = torch.sum((rewards - q_values)**2)

        #print(delta)
    # Backprop
    for j in range(len(log_probs)):
        if log_probs[j]:
            if len(losses) < num_agents:
                losses.append(torch.cat(log_probs[j]).sum())
            else:
                losses[j] = torch.cat(log_probs[j]).sum()
            losses[j] *= delta[j].clone().detach()
    if (i+1) % batch_size == 0:

        optimizer_critic.zero_grad()
        optimizer_q_critic.zero_grad()
        loss_critic.backward()
        loss_q_critic.backward()
        optimizer_critic.step()
        optimizer_q_critic.step()
        critic.hidden = None
        q_critic.hidden = None
        for j, loss in enumerate(losses):
            loss = -loss
            optimizers[j].zero_grad()
            loss.backward()
            optimizers[j].step()
        for agent in agents:
            agent.hidden = np.empty(num_agents, dtype=tuple)
        losses = []
    if (i+1) % 100 == 0:
        print((i+1), rewards, delta)
        # print("Utterance", game.state.last_utterance, "Hidden utils", game.state.hidden_utils[game.state.curr_player])
    # rewards_saved[i] = rewards
    rewards_saved[i] = rewards
torch.save(agents[0], "agent0_self")


rewards_saved = np.array(rewards_saved.cpu())
#print(rewards_saved)
pos_reward = reverse_order(rewards_saved.copy(), starting_player)
#print(rewards_saved)
plot(rewards_saved, pos_reward)

#print(state.item_pool, state.hidden_utils, state.last_proposal, rewards, curr_player)