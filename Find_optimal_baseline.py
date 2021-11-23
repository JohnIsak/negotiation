import numpy as np
import Negotiation

avg_reward = np.zeros(2)
for i in range(100_000):
    game = Negotiation.NegotiationGame()
    best_proposal = game.find_best_solution()
    _, _ = game.apply_action(best_proposal, None, False)
    _, reward = game.apply_action(np.zeros(3), None, True)
    avg_reward += reward

avg_reward /= 100_000
print(avg_reward)
