import numpy as np
import Negotiation_continuous as Negotiation_continuous
import Reinforce_agent_LSTM
import Critic
import torch
import matplotlib.pyplot as plt
import pandas as pd

torch.autograd.set_detect_anomaly(True)


def reverse_order(rewards_saved, starting_player):
    for i in range(len(rewards_saved)):
        rewards_saved[i] = rewards_saved[i, 0] if starting_player[i] == 0 else rewards_saved[i, 1]
    return rewards_saved


def plot(rewards_saved, pos_reward):
    avg = pd.DataFrame(rewards_saved[:, 0])
    avg = avg.iloc[:, 0].rolling(window=10).mean()

    avg_2 = pd.DataFrame((rewards_saved[:, 0]+ rewards_saved[:, 1])/2)
    avg_2 = avg_2.iloc[:, 0].rolling(window=10).mean()

    max_min = np.sort(rewards_saved, 1)
    max_min = pd.DataFrame(max_min)

    max_min_plot = []
    for j in range(num_agents):
        max_min_plot.append(max_min.iloc[:, j].rolling(window=10).mean())

    rewards = pd.DataFrame(pos_reward)

    players = []
    for j in range(num_agents):
        players.append(rewards.iloc[:, j].rolling(window=10).mean())

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
agents = [Reinforce_agent_LSTM.Reinforce_agent(), Reinforce_agent_LSTM.Reinforce_agent()]
critic = Critic.Critic()
critic = critic.to(device=device)
#q_critic = Critic.Critic(13+7)
#q_critic = q_critic.to(device=device)

agents[0] = agents[0].to(device=device)
agents[1] = agents[1].to(device=device)
optimizers = [torch.optim.Adam(agents[0].parameters(), lr=0.001),
              torch.optim.Adam(agents[1].parameters(), lr=0.001)]

optimizer_critic = torch.optim.Adam(critic.parameters(), lr=0.001)
# optimizer_q_critic = torch.optim.Adam(q_critic.parameters(), lr=0.0001)x
torch.autograd.set_detect_anomaly(True)
batch_size = 128
still_alive = []
num_iterations = 10_000

starting_player = torch.zeros(num_iterations)
rewards_saved = np.zeros((num_iterations, num_agents))
max_reward_prosocial = torch.zeros((num_iterations, batch_size))
losses = []
loss_critic = torch.tensor(0.0, device=device)
rewards_tot_old = 0

torch.save(agents[0], "Marl0")
torch.save(agents[1], "Marl1")
torch.save(critic, "Critic")
for i in range(num_iterations):
    # Play an episode
    game = Negotiation_continuous.NegotiationGame(batch_size)
    state = game.state
    starting_player[i] = state.curr_player
    rewards = torch.zeros((batch_size, 2), device=device)

    log_probs = [torch.zeros(batch_size, device=device), torch.zeros(batch_size, device=device)]
    v = torch.zeros((batch_size, 2), device=device)

    while sum(state.still_alive) != 0:
        state_coded = state.generate_processed_state()
        still_alive.append(state.still_alive.clone().detach())
        agreement, proposal, log_prob, utterance = agents[state.curr_player].act(state_coded, still_alive[-1])

        v_est = critic(state_coded, still_alive[-1])
        v[still_alive[-1], state.curr_player] = torch.reshape(v_est, (-1,))

        log_probs[state.curr_player][still_alive[-1]] += torch.sum(log_prob, dim=1)
        state, rewards = game.apply_action(proposal, utterance, agreement, rewards)

    delta = rewards - v
    loss_critic = torch.sum(-delta.clone().detach() * v)/batch_size
    #print(loss_critic)

    # Backprop
    for j, log_prob in enumerate(log_probs):
        losses.append(torch.sum(log_prob * delta[:,j].clone().detach()))
        losses[-1] = losses[-1]/batch_size

    if (i+1) % 1 == 0:
        optimizer_critic.zero_grad()
        loss_critic.backward()
        optimizer_critic.step()
        critic.h_n = None
        critic.c_n = None
        loss_critic = torch.tensor(0.0, device=device)
        for j, loss in enumerate(losses):
            loss = -loss
            optimizers[j].zero_grad()
            loss.backward()
            optimizers[j].step()
        for agent in agents:
            agent.h_n = None
            agent.c_n = None
        losses = []
    if (i+1) % 100 == 0:
        print((i + 1), torch.sum(rewards, dim=0)/batch_size, torch.sum(delta, dim=0)/batch_size)
    if (i+1) % 999999999999 == 0:
        rewards_tot = 0
        for j in range(1000):
            game = Negotiation_continuous.NegotiationGame(num_agents)
            state = game.state
            v = torch.empty(2, device=device)
            while not state.is_terminal:
                state_coded = state.generate_processed_state()
                agreement, proposal, log_prob, utterance = agents[state.curr_player].act(state_coded, state.curr_player)
                v[state.curr_player] = critic(state_coded)
                log_probs[state.curr_player].append(log_prob)
                state, rewards = game.apply_action(proposal, utterance, agreement)

            rewards = rewards / game.find_rewards([game.best_collective_proposal()])
            rewards -= 0.01 * state.turn
            rewards_tot += rewards[0]
            for agent in agents:
                agent.hidden = np.empty(num_agents, dtype=tuple)
            critic.hidden = None
        if rewards_tot > rewards_tot_old:
            rewards_tot_old = rewards_tot
            torch.save(agents[0], "Marl0")
            torch.save(agents[1], "Marl1")
            torch.save(critic, "Critic")
        else:
            agents[0] = torch.load("Marl0")
            agents[1] = torch.load("Marl1")
            critic = torch.load("Critic")


    # print(rewards.shape, "Rewards shape")
    # print((torch.sum(rewards, dim=0)/batch_size).cpu(), "summed shape")
    rewards_saved[i] = (torch.sum(rewards, dim=0)/batch_size).cpu().numpy()
torch.save(agents[0], "agent0_self")


pos_reward = reverse_order(rewards_saved.copy(), starting_player)
plot(rewards_saved, pos_reward)

#print(state.item_pool, state.hidden_utils, state.last_proposal, rewards, curr_player)