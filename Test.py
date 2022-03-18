import torch
import numpy as np
import pandas as pd
import Reinforce_agent_LSTM as reinforce_agent
import Negotiation_continuous as Negotiation

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#GENERATING STATE TEST
gamez = Negotiation.NegotiationGame(6)
state = gamez.state
state, still_alive = state.generate_processed_state()
#print(state)


#FIND REWARDS TEST::
batch_size = 6
agreement = torch.tensor([0, 0, 1, 1, 1, 0], dtype=torch.bool, device=device)
proposals = torch.tensor([[1,1,1],[1,1,0],[1,0,0],[0,0,0],[0.5,0.5,0.5],[0.5,0.5,1]], device=device)
state.proposals = proposals
rewards = torch.zeros((batch_size, 2), device=device)
rewards = gamez.find_rewards(agreement, rewards)
# print(rewards)

#APPLY ACTION TEST
rewards = torch.zeros((batch_size, 2), device=device)
agreement = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.bool, device=device)
proposals = torch.tensor([[1,1,1],[1,1,0],[1,0,0],[0,0,0],[0.5,0.5,0.5],[0.5,0.5,1]], device=device)
utterances = torch.tensor([[0,0.1,0.2],[1,1,0],[1,0,0],[0,0,0],[0.5,0.5,0.5],[0.5,0.5,1]], device=device)
state, rewards, still_alive = gamez.apply_action(proposals, utterances, agreement, rewards)
print(still_alive)
print(rewards)
agreement = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool, device=device)
proposals = torch.tensor([[1,1,1],[1,1,0],[1,0,0],[0,0,0],[0.5,0.5,0.5],[0.5,0.5,1]], device=device)
state, rewards, still_alive = gamez.apply_action(proposals, utterances, agreement, rewards)
print(still_alive)
print(rewards)


#CHANGING DIFFERENT TERMINATION POINTS TEST
gamez = Negotiation.NegotiationGame(batch_size)
state = gamez.state
state, still_alive = state.generate_processed_state()
rewards = torch.zeros((batch_size, 2), device=device)

agreement = torch.tensor([0, 0, 0, 0, 1, 0], dtype=torch.bool, device=device)
proposals = torch.tensor([[1,1,1],[1,1,0],[1,0,0],[0,0,0],[0.5,0.5,0.5],[0.5,0.5,1]], device=device)
utterances = torch.tensor([[0,0.1,0.2],[1,1,0],[1,0,0],[0,0,0],[0.5,0.5,0.5],[0.5,0.5,1]], device=device)
state, rewards, still_alive = gamez.apply_action(proposals, utterances, agreement, rewards)
print(still_alive)
print(rewards)
print(state)
agreement = torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.bool, device=device)
proposals = torch.tensor([[0.5,0.5,0.5],[0.5,0.5,1],[0.5,1,1],[1,1,1],[0,0,0],[0,0,0.5]], device=device)
utterances = torch.tensor([[0,0.1,0.2],[1,1,0],[1,0,0],[0,0,0],[0.5,0.5,0.5],[0.5,0.5,1]], device=device)
state, rewards, still_alive = gamez.apply_action(proposals, utterances, agreement, rewards)
print(still_alive)
print(state)
print(rewards)
agreement = torch.tensor([1, 1],dtype=torch.bool, device=device)
proposals = torch.tensor([[0.5,0.5,0.5],[0.5,0.5,1]], device=device)
utterances = torch.tensor([[0.5,0.5,0.5],[0.5,0.5,1]], device=device)
state, rewards, still_alive = gamez.apply_action(proposals, utterances, agreement, rewards)
print(still_alive)
print(state)
print(rewards)

#AGENT LSTM BATCHING TEST
a = reinforce_agent.Reinforce_agent()
nums = torch.rand(3,1,13)
mask = torch.tensor([True, True, True], dtype=torch.bool)
out = a.act(nums, mask)
#print(a.h_n)
#print(a.h_n.shape) # Hidden 0 har shape
#print(out)
mask = torch.tensor([True, False, True], dtype=torch.bool)
out = a.act(nums, mask)
#print(out)


#print(a.hidden[1])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_players = 3

utils_generated = False
while not utils_generated:
    hidden_utils = torch.randint(0, 11, (num_players, 3), device=device)
    hidden_utils[0,0] = 0
    utils_generated = True
    for i in range(num_players):
        if sum(hidden_utils[i]) == 0:
            utils_generated = False
print(hidden_utils)


a = torch.tensor([1,2,3])
b = torch.tensor([4])
c = torch.cat((a,b))
#print(c)

a = np.random.random((100,2))
#print(a)
a = np.sort(a, axis=1)
#print(a)
a = pd.DataFrame(a)
#print(a)


m = torch.distributions.Categorical(torch.tensor([0.5, 0.25, 0.25]))
a = m.sample()

#print(a)
#print(m.log_prob(a))

mean = torch.tensor([0.2, 0.5, 0.4])
std = torch.tensor([0.5, 0.3, 0.1])
distributions = torch.distributions.normal.Normal(mean, std) #STD should be positive

a = distributions.sample()

log_prob = distributions.log_prob(a)
log_prob_2 = (-(a-mean)**2)/(2*std**2)
log_prob_2 = log_prob_2-torch.log(np.sqrt(2*np.pi*std**2))

a = a.detach().numpy()

#print(distributions)
#print(a)
#print(log_prob)
#print(log_prob_2)
