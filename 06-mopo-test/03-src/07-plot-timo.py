import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("../04-results/01-data/results_epochs(2).csv")
epoch = 199
depth = 4
df = df[df["epoch"].astype(int) == epoch]
df = df[df["depth"] == depth]

plt.title(f"epoch={epoch}, d={depth}, Width vs Loss")
plt.plot(df["width"], df["loss_test"])
plt.ylabel("test_loss")
plt.xlabel("width_param")
plt.show()
