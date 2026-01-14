from classes.utils import load_data
from pathlib import Path
from classes.network import Network
from pathlib import Path
import csv

if __name__ == "__main__":
    data_dir = Path("../01-data/02-preprocessed")
    x_train, x_test, y_train, y_test = load_data(data_dir)

    depth = 4
    widths = [i for i in range(10, 41, 30)]
    epochs = 100
    step = 10

    out_path = Path("results_epochs.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["depth", "width", "epoch", "loss_train", "loss_test"])
        for width in widths:
            net = Network(h_layers=depth, h_size=width)
            for ep in range(step,epochs, step):
                print(f"width:{width}, depth:{depth}")
                # train_losses = net.train(..., return_history=True)
                # If your train returns nothing, you must log manually (see below)
                train_losses = net.train(x_train, y_train, epochs=step)
                
                # If train() returns a list with one entry per epoch:
                loss_train = float(train_losses[-1])
                loss_test = float(net.mse_loss(x_test, y_test))
                writer.writerow([depth, width, ep, loss_train, loss_test])

    print(f"Saved: {out_path}")