import torch


mean = torch.tensor([0.2, 0.5, 0.4])
std = torch.tensor([0.5, 0.3, 0.1])
distributions = torch.distributions.normal.Normal(mean, std) #STD should be positive

a = distributions.sample()
log_prob = distributions.log_prob(a)
print(distributions)
print(a)
print(log_prob)
