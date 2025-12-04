import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def plot_arrays(losses, sweep):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.plot(sweep, losses)
    ax.set_title("loss over cap.")
    ax.set_ylabel("loss")
    ax.set_xlabel("cap.")
    plt.show()

class Network(nn.Module):
    def __init__(self, in_dim=5, h_dim=64, out_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h_dim) # -> s, a
        self.h1 = nn.Linear(h_dim, h_dim)
        # self.h2 = nn.Linear(64, 64)
        # self.h3 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(h_dim, out_dim) # -> s_new, r

    def forward(self, input):
        res = torch.relu(self.fc1(input))
        res = torch.relu(self.h1(res))
        res = self.fc2(res)
        return res

def model_run(in_dim, h_dim, out_dim, x_train, y_train, x_test, y_test):
    model = Network(in_dim, h_dim, out_dim)
    criterion = nn.MSELoss()        # or nn.CrossEntropyLoss if classification
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = x_train
    targets = y_train

    for epoch in range(1000):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss = {loss.item():.4f}")
    
    outputs = model(x_test)
    loss = criterion(outputs, y_test)
    print(f"test loss: {loss.item()}")
    return loss.item()

sweep =[64, 128, 256, 512, 1024]

data = np.load("data.npz")
inputs = torch.tensor(data['inputs'])
targets = torch.tensor(data['targets'])

x_train, x_test, y_train, y_test = train_test_split(
    inputs,
    targets,
    test_size=0.2,      # 20% test
    random_state=42,    # reproducible split
    shuffle=True,
)

arrays = []

for val in sweep:
    print("sweep...")
    arrays.append(model_run(5, val, 5, x_train, y_train, x_test, y_test)) # 5, new hidden, 5

plot_arrays(arrays, sweep)