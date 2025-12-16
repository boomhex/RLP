from .network import Network

import torch

class Ensemble:

    def __init__(self, n_networks, lmbda=0.5, hidden_size=10, hidden_layers=4):

        self.n_networks = n_networks
        self.models = [
            Network(h_size=hidden_size, h_layers=hidden_layers) for i in range(n_networks)
        ]
        self.lmbda = lmbda

    def train(self, x, y, epochs=500):
        for model in self.models:
            model.train(x, y, epochs=500)

    def test(self, x, y):
        predictions: torch.Tensor = self.predict(x)    # get the prediction
        loss = self.mse(predictions, y)                # calculate the loss
        return loss                                         # report the loss

    def forward(self, x) -> torch.Tensor:
        predictions: torch.Tensor = torch.stack(           # shape: (self.n_models, 2, N, out_dim)
            [model.forward(x) for model in self.models]
        )
        return predictions

    def predict(self, x) -> torch.Tensor:
        predictions: torch.Tensor = self.forward(x)               # forward for all models

        # process ensemble predictions and conclude on 1 prediction
        corrected_prediction: torch.Tensor = self.process_predictions(predictions)

        return corrected_prediction

    def process_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        # separate means and variances
        means: torch.Tensor = predictions[:, 0, :, :]
        vars: torch.Tensor = predictions[:, 1, :, :]

        # randomly select means
        random_mean = self.select_random_mean(means)

        # calculate max variance
        max_var = self.max_variance(vars)

        # correct reward prediction using max variance
        corrected_prediction = self.penalize_prediction(random_mean, max_var)

        return corrected_prediction

    def select_random_mean(self, means: torch.Tensor):
        # means of shape: (7, N, 18)
        n_nets, n_samples, n_dims = means.shape

        # random network index per sample: (N,)
        idx = torch.randint(
            low=0,
            high=n_nets,
            size=(n_samples, 1),
            device=means.device
        )
        # reshape for gather: (N, 7, 18)
        means_n = torch.permute(means, (1, 0, 2))

        # gather expects index shape to match output shape
        # -> (N, 1, 18), then squeeze to (N, 18)
        idx_g = idx.view(n_samples, 1, 1).expand(-1, 1, n_dims)
        out = means_n.gather(dim=1, index=idx_g).squeeze(1)

        return out
    
    def max_variance(self, vars: torch.Tensor) -> torch.Tensor:
        out = torch.max(vars[:, :, -1], dim=0).values
        return out

    def penalize_prediction(self,
                            mean: torch.Tensor,
                            max_var: torch.Tensor):
        mean[:, -1] = mean[:, -1] - self.lmbda * max_var
        return mean
    
    def mse(self, y: torch.Tensor, y_hat: torch.Tensor):
        mse = (y - y_hat)**2        # mse per prediction
        return torch.mean( mse )    # mean mse

    def to(self, device):
        for model in self.models:                         # move each model to device
            model.to(device)