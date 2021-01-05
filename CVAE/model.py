import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self,
                 input_dim: int,
                 cond_dim: int,
                 latent_dim: int,
                 reg: Tensor = torch.ones(1),
                 bias: bool = True,
                 layers: list = None,
                 device: torch.device = torch.device("cpu"),
                 **kwargs) -> None:
        super().__init__()

        self.device = device
        self.latent_dim = latent_dim
        self.reg = reg.to(self.device)

        # encoder
        modules = []
        in_features = input_dim + cond_dim
        for h_dim in layers:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features, h_dim, bias),
                    nn.ReLU()),
            )
            in_features = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(layers[-1], latent_dim)
        self.fc_logvar = nn.Linear(layers[-1], latent_dim)

        #  decoder
        modules = []
        in_features = latent_dim + cond_dim
        for h_dim in reversed(layers):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features, h_dim, bias),
                    nn.ReLU()
            ))
            in_features = h_dim

        modules.append( # final layer is linear
            nn.Linear(in_features, input_dim, bias)
        )
        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor, cond: Tensor) -> list:
        """
        encodes the input and returns the code pair [mu, logvar]
        """
        try:
            x = torch.cat([input, cond], dim=1)
        except:
            if len(input.shape) == 1:
                input = input.unsqueeze(1)
            if len(cond.shape) == 1:
                cond = cond.unsqueeze(1)
            x = torch.cat([input, cond], dim=1)
        code = self.encoder(x)
        mu = self.fc_mu(code)
        logvar = self.fc_logvar(code)
        return [mu, logvar]

    def sample_latent(self, mu:Tensor, logvar:Tensor) -> Tensor:
        """
        given the code pair returns latent rv. z
        """
        sdev = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sdev, device = self.device)
        return (mu + eps * sdev)

    def decode(self, z: Tensor, cond: Tensor) -> Tensor:
        """
        decodes the latent rv. z given conditional knowledge
        """
        try:
            x = torch.cat([z, cond], dim=1)
        except:
            if len(z.shape) == 1:
                z = z.unsqueeze(1)
            if len(cond.shape) == 1:
                cond = cond.unsqueeze(1)
            x = torch.cat([z,cond], dim=1)
        return self.decoder(x)

    def forward(self, input:Tensor, cond:Tensor) -> list:
        [mu, logvar] = self.encode(input, cond)
        z = self.sample_latent(mu, logvar)
        return [self.decode(z, cond), mu, logvar]

    def loss_func(self,
                  pred: Tensor,
                  input: Tensor,
                  mu: Tensor,
                  logvar: Tensor) -> list:
        MSE = F.mse_loss(pred, input)
        try:
            KLD = torch.mean(-0.5 * torch.sum(1+logvar - mu.pow(2) - logvar.exp(), dim=1))
        except:
            KLD = torch.mean(-0.5 * 1+logvar - mu.pow(2) - logvar.exp() )

        return [MSE*self.reg + KLD, MSE, KLD]

    def sample_trajectories(self,
                            num_samples: int,
                            cond: Tensor) -> list   :

        z = torch.randn(num_samples, self.latent_dim)
        c = torch.stack([cond for _ in range(num_samples)], dim=0)
        return self.decode(z, c).detach()
