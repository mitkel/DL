import torch
import torch.nn.functional as nn

class CVAE(nn.Module):
    def __init__(self,
                 input_dim: int,
                 cond_dim: int,
                 latent_dim: int,
                 bias: bool = True,
                 layers: List = None,
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        # encoder
        in_features = input_dim + cond_dim
        modules = []

        for h_dim in layers:
            modules.append(
                nn.Linear(in_features=in_features,
                          out_features=h_dim,
                          bias=bias),
                nn.ReLU,
            )
            in_features = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(layers[-1], latent_dim)
        self.fc_logvar = nn.Linear(layers[-1], latent_dim)


        modules = []
        in_features = latent_dim + cond_dim
        for h_dim in list(reversed(layers)):
            modules.append(
                nn.Linear(in_features=in_features,
                          out_features=h_dim,
                          bias=bias),
                nn.ReLU,
            )
            in_features = h_dim

        modules.append( # final layers is linear
            nn.Linear(in_features=in_features,
                      out_features=input_dim,
                      bias=bias)
        )
        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List(Tensor):
        """
        encodes the input and returns code pair [mu, logvar]
        """
        code = self.encoder(input)
        mu = self.fc_mu(code)
        logvar = self.fc_logvar(code)
        return [mu, logvar]

    def decode(self, z: Tensor, cond: Tensor) -> Tensor:
        x = torch.cat([z, cond])
        return self.decoder(x)

    def sample_latent(self, mu:Tensor, logvar:Tensor) -> Tensor:
        """
        given code pair returns latent variable
        """
        sdev = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sdev)
        return mu + eps * sdev

    def forward(self, input:Tensor, cond:Tensor) -> List(Tensor):
        x = torch.cat([input, cond], dim=1)
        [mu, logvar] = self.encode(x)
        z = self.sample_latent(mu, logvar)
        return [self.decode(z), mu, logvar]

    def sample_trajectories(self,
                            num_samples: int,
                            cond: Tensor) -> List(Tensor):

        z = torch.randn(num_samples, self.latent_dim)
        return [self.decode(z[i,:], cond) for i in range(num_samples)]
