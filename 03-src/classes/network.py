import torch
from torch import nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Network(nn.Module):   # class defining a basic nn

    def __init__(self, h_size=200, h_layers=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(23, h_size),      # in
            nn.ReLU(),
        )
        for i in range(h_layers):
            self.model.append(nn.Linear(h_size, h_size))
            self.model.append(nn.ReLU())
        self.mean_head = nn.Linear(h_size, 18)
        self.logvar_head = nn.Linear(h_size, 18)

        # bind log-variance to avoid numerical instability
        self.max_logvar = nn.Parameter(torch.ones(18) * 0.5)
        self.min_logvar = nn.Parameter(torch.ones(18) * -10)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()    # using mean squared error as a loss metric

    def forward(self, x) -> torch.Tensor:
        res = self.model(x)
        mean = self.mean_head(res)
        logvar = self.logvar_head(res)

        # clamp log-variance using soft constraints (see MBPO/PETS)
        logvar = self.max_logvar - torch.nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + torch.nn.functional.softplus(logvar - self.min_logvar)

        return torch.stack([mean, logvar])

    def nll_loss(self, x, y):
        """
        Negative log-likelihood of Gaussian:
            NLL = 0.5 * [ logσ² + (y - µ)² / σ² ]
        """
        mean, logvar = self.forward(x)
        var = torch.exp(logvar)

        nll = 0.5 * ((y - mean)**2 / var + logvar)
        return nll.mean()

    def train_epoch(self, x, y):
        self.optimizer.zero_grad()

        loss = self.nll_loss(x, y)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, x, y, epochs=500):
        # again split the data to optimize hyperparam on val set, to not leak data.
        # x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=True)
    
        losses = []
    
        for iter in tqdm(range(epochs)):
            # train
            iteration_loss = self.train_epoch(x, y)
            losses.append(iteration_loss)

        return losses

    def mse_loss(self, x, y):
        y_hat = self.forward(x)[0]
        mse = torch.mean((y_hat - y)**2)
        return mse

    def validation_loss(self, x, y):
        loss = self.nll_loss(x, y)
        return loss.item()

    def reset(self):
        self.__init__()