import numpy as np
import matplotlib.pyplot as plt

data = np.load("data.npz")
rewards = data["targets"][0]

print(rewards)
plt.plot(rewards)
plt.show()