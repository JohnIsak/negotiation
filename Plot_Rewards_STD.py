import numpy as np
import seaborn
import matplotlib.pyplot as plt
import pandas as pd

print(np.tile(np.arange(0,10),3))
PL = np.load("Rewards Positive Listening.npy").mean(1)
print(PL.shape)
PL = np.tile(PL, 6)
print(PL.shape)
PL[100_000:200_000] = np.load("Rewards Positive Signalling 2.npy").mean(1)
PL[200_000:300_000] = np.load("Rewards Positive Signalling 3.npy").mean(1)
PL[300_000:400_000] = np.load("Rewards No Positive Listening.npy").mean(1)
PL[400_000:500_000] = np.load("Rewards No Positive Signalling 2.npy").mean(1)
PL[500_000:600_000] = np.load("Rewards No Positive Signalling 3.npy").mean(1)

data = pd.DataFrame(PL, columns=["Reward"])
data["Iteration"] = np.tile(np.arange(0,100_000), 6)
positive_Signalling_mask = np.repeat(1, 600_000)
positive_Signalling_mask[300_000:600_000] = 0
data["Positive Signalling"] = pd.Categorical(np.where(positive_Signalling_mask, "With", "Without"))

# data.rename(columns={"0":"vals"}, inplace=True)

print(data)

# print(PL.shape)
# print(PL[:,0:100].shape)
print(data[0:100].append(data[300_000:300_100]))

seaborn.set()
seaborn.lineplot(data=data[0:1000].append(data[300_000:301_000]), x="Iteration", y="Reward", hue="Positive Signalling",
                 style="Positive Signalling", dashes=True)
plt.show()

