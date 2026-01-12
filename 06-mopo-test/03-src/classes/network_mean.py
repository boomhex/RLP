import torch
from torch import nn
from tqdm import tqdm

class Network(nn.Module):   # class defining a basic nn

    def __init__(self, h_size=200, h_layers=4):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(23, h_size),      # in
            nn.ReLU(),
        )
        for i in range(h_layers):
            self.model.append(nn.Linear(h_size, h_size))
            self.model.append(nn.ReLU())

        self.mean_head = nn.Linear(h_size, 18)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()    # using mean squared error as a loss metric
        self.to(self.device)

    def forward(self, x) -> torch.Tensor:
        x = x.to(self.device)
        res = self.model(x)
        mean = self.mean_head(res)

        return mean

    def train_epoch(self, x, y):
        self.optimizer.zero_grad()

        loss = self.mse_loss(x, y)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def mse_loss_nograd(self, x, y):
        y_hat = self.forward(x)          # (B, 18)
        return torch.mean((y_hat - y) ** 2)
    
    def mse_loss(self, x, y):
        y_hat = self.forward(x)          # (B, 18)
        return torch.mean((y_hat - y) ** 2)

    def validation_loss(self, x, y):
        loss = self.nll_loss(x, y)
        return loss.item()

    def reset(self):
        self.__init__()