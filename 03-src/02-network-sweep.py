from classes.utils import load_data
from pathlib import Path
from classes.network import Network

if __name__ == "__main__":
    # load data
    data_dir = Path("../01-data/02-processed")
    x_train, x_test, y_train, y_test = load_data(data_dir)

    depths = [2, 3, 4, 5, 6]
    widths = [2**i for i in range(1, 5)]        # --> [2, 4, 8, 16,  .., 2^i]

    results = {}
    for depth in depths:
        results[depth] = {}
        for width in widths:
            net = Network(h_layers=depth, h_size=width)
            net.train(x_train, y_train, epochs=100)
            mse_loss = net.mse_loss(x_test, y_test)
            results[depth][width] = mse_loss
    
    


