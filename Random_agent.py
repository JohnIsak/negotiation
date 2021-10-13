import numpy as np


class RandomAgent:

    def __init__(self, item_pool):
        self.item_pool = item_pool

    def act(self):
        proposal = np.repeat(0, 3)
        for i in range(3):
            proposal[i] = np.random.randint(0, self.item_pool[i] + 1)
        agreement = 0
        if np.random.random() <= 0.1:
            agreement = 1
        return proposal, agreement
