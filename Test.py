import torch
import numpy as np
import pandas as pd

a = torch.tensor([1,2,3])
b = torch.tensor([4])
c = torch.cat((a,b))
#print(c)

a = np.random.random((100,2))
print(a)
a = np.sort(a, axis=1)
print(a)
a = pd.DataFrame(a)
print(a)


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
