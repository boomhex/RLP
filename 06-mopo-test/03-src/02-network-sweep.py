from classes.utils import load_data
from pathlib import Path
import torch
import numpy as np
from classes.network_mean import Network
import csv
from tqdm import tqdm


def savetofile(file, results):
    with open(file, 'w') as f:
        f.write(f"{results}\n")

def permute_y_fraction(y: torch.Tensor, frac: float, generator=None) -> torch.Tensor:
    """
    Returns a copy of y where a fraction `frac` of rows are replaced by
    randomly permuted rows (label permutation noise).

    Assumes y is shaped (N, ...) and you want to permute along the first dim.
    """
    if not (0.0 <= frac <= 1.0):
        raise ValueError("frac must be in [0, 1]")

    y_noisy = y.clone()
    n = y_noisy.shape[0]
    k = int(round(frac * n))
    if k == 0:
        return y_noisy

    if generator is None:
        generator = torch.Generator(device=y.device)

    idx = torch.randperm(n, generator=generator, device=y.device)[:k]
    perm = torch.randperm(n, generator=generator, device=y.device)
    y_noisy[idx] = y_noisy[perm[idx]]
    return y


if __name__ == "__main__":
    # load data
    data_dir = Path("../01-data")
    x_train, x_test, y_train, y_test = load_data(data_dir)

    seed = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    noise_frac = 0.2  # 20% of y rows become wrong (permuted)
    y_train = permute_y_fraction(y_train, noise_frac)

    x_train = x_train.to(device)
    x_test = x_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    samples_train = 10000
    test_frac = 0.25
    samples_test = int(samples_train * test_frac)

    x_train = x_train[:samples_train]
    y_train = y_train[:samples_train]

    x_test = x_test[:samples_test]
    y_test = y_test[:samples_test]

    depth = 4
    widths = [i for i in range(50, 1000, 10)]
    epochs = 600
    step = 1

    out_path = Path("./results_epochs.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["depth", "width", "epoch", "loss_train", "loss_test", "net_loss"])
        for width in widths:

            net = Network(h_layers=depth, h_size=width)
            print(f"width:{width}, depth:{depth}")
            for ep in range(step, epochs, step):

                train_losses = net.train_epoch(x_train, y_train)

                loss_train = float(train_losses)
                loss_test = float(net.mse_loss_nograd(x_test, y_test))

                writer.writerow([depth, width, ep, loss_train, loss_test, train_losses])

    print(f"Saved: {out_path.absolute()}")