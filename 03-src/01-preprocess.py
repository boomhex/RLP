import h5py
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import torch

if __name__ == "__main__":
    # define paths
    data_base = Path("../01-data/")
    raw_data = Path("01-raw/halfcheetah_medium-v2")
    destination = Path("02-preprocessed")

    data = h5py.File(Path("./../01-data/01-raw/halfcheetah_medium-v2.hdf5"))

    # extract relevant cols
    actions = data["actions"]
    state_new = data["next_observations"]
    state = data["observations"]
    rewards = data["rewards"]

    # info
    print(
        f"a shape = {actions.shape}\n" \
        f"s shape = {state.shape}\n" \
        f"s_new shape = {state_new.shape}\n" \
        f"r shape = {rewards.shape}\n"
    )

    # divide data
    x = np.hstack([actions, state])                                # -> (N, 23)
    y = np.hstack([state_new, np.array(rewards).reshape(-1, 1)])   # -> (N, 18)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # converting to tensors
    x = torch.tensor(x, dtype=torch.float32).to(device)   
    y = torch.tensor(y, dtype=torch.float32).to(device)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True
    )

    # info
    print(
        f"x_train shape = {x_train.shape}\n" \
        f"x_test shape = {x_test.shape}\n" \
        f"y_train shape = {y_train.shape}\n" \
        f"y_test shape = {y_test.shape}"
    )

    if not Path.exists(data_base/destination):
        Path.mkdir(data_base/destination)
    
    torch.save(x_train, data_base/destination/"x_train.pt")
    torch.save(y_train, data_base/destination/"y_train.pt")
    torch.save(x_test, data_base/destination/"x_test.pt")
    torch.save(y_test, data_base/destination/"y_test.pt")

