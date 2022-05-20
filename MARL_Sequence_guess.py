import numpy as np
import Sequence_guess
import Seq_Agent_LSTM
import torch
import matplotlib.pyplot as plt
import pandas as pd

torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def reverse_order(rewards_saved, starting_player):
    for i in range(len(rewards_saved)):
        rewards_saved[i] = rewards_saved[i, 0] if starting_player[i] == 0 else rewards_saved[i, 1]
    return rewards_saved


def plot(rewards_saved, num_agents):
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



def main():

    baseline = 0.5
    guess_agent = Seq_Agent_LSTM.Reinforce_agent(2, 2, 2)
    guess_agent = guess_agent.to(device)
    mastermind_agent = Seq_Agent_LSTM.Reinforce_agent(2 * 2, 2, 2)
    mastermind_agent.to(device)

    optimizers = [torch.optim.Adam(guess_agent.parameters(), lr=0.001),
                  torch.optim.Adam(mastermind_agent.parameters(), lr=0.001)]
    batch_size = 2048
    num_iterations = 100_000
    rewards_saved = np.zeros((num_iterations, 2))
    losses = []
    rewards_tot_old = torch.zeros(2)
    for i in range(num_iterations):
        # Play an episode
        log_probs, rewards, = play_episode(mastermind_agent, guess_agent, batch_size)
        delta = rewards - baseline
        baseline = 0.7*baseline + 0.3*rewards[0]
        # Backprop
        for j, log_prob in enumerate(log_probs):
            losses.append(torch.sum(log_prob * delta[:, j].clone().detach()))
            losses[-1] = losses[-1] / batch_size

        if (i + 1) % 1 == 0:
            for j, loss in enumerate(losses):
                loss = -loss
                optimizers[j].zero_grad()
                loss.backward()
                optimizers[j].step()
            losses = []
        if (i + 1) % 10 == 0:
            print((i + 1), torch.sum(rewards, dim=0) / batch_size, torch.sum(delta, dim=0) / batch_size,
                  "many channels")
            _, rewards_test = play_episode(mastermind_agent, guess_agent, batch_size, True)
            rewards_tot = torch.sum(rewards_test, dim=0)/batch_size
            print(rewards_tot, "Reward, testing")
            if torch.sum(rewards_tot) > torch.sum(rewards_tot_old):
                rewards_tot_old = rewards_tot
                torch.save(guess_agent, "Guess Agent")
                torch.save(mastermind_agent, "Mastermind Agent")

        rewards_saved[i] = (torch.sum(rewards, dim=0) / batch_size).cpu().numpy()
    plot(rewards_saved, 2)


def play_episode(mastermind, guesser, batch_size, testing=False):
    game = Sequence_guess.SequenceGame(batch_size, 2, 2, 2, 2)
    state = game.state
    rewards = torch.zeros((batch_size, 2), device=device)
    log_probs = [torch.zeros(batch_size, device=device), torch.zeros(batch_size, device=device)]
    while torch.sum(state.still_alive) != 0:
        input = state.generate_processed_state()[state.still_alive.clone()]
        guess, log_prob = guesser.forward(input)
        if not testing:
            log_probs[state.curr_player][state.still_alive.clone()] += torch.sum(log_prob, dim=1)
        state, rewards = game.apply_action(rewards, guess=guess)

        input = state.generate_processed_state()[state.still_alive.clone()]
        utt, log_prob = mastermind.forward(input)
        if not testing:
            log_probs[state.curr_player][state.still_alive.clone()] += torch.sum(log_prob, dim=1)
        state, rewards = game.apply_action(rewards, utt=utt)

    return log_probs, rewards


main()

# print(state.item_pool, state.hidden_utils, state.last_proposal, rewards, curr_player)