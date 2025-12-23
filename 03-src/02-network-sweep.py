from classes.utils import load_data
from pathlib import Path
from classes.network import Network
import json
from typing import Dict

def save(dic: Dict, fp: Path) -> None:
    with open(fp, 'w') as file:
        json.dump(dic, file, indent=True)

if __name__ == "__main__":
    # load data
    data_dir = Path("../01-data/02-preprocessed")
    save_path = Path("../05-results/data")

    x_train, x_test, y_train, y_test = load_data(data_dir)

    depths = [2, 3]
    widths = [i for i in range(1, 2)]

    results = {}
    for depth in depths:

        results[depth] = {}

        for width in widths:

            net = Network(h_layers=depth, h_size=width)
            net.train(x_train, y_train, epochs=400)
            mse_loss = net.mse_loss(x_test, y_test)
            results[depth][width] = mse_loss.item()

            save(results, save_path/"results.json")
