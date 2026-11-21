import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim: int=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features=28*28, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc_mu = nn.Linear(256, latent_dim)  # project to mean
        self.fc_logvar = nn.Linear(256, latent_dim)  # project to log-variance
        
    def forward(self, x: torch.Tensor):
        # x shape [B, C, H, W]
        # flatten it to [B, C*H*W]
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

def reparameterise(mu, logvar):
    std = torch.exp(0.5 * logvar)  # convert log variance to standard deviation
    eps = torch.randn_like(std)  # sample from standard Gaussian (N(0,1))
    return mu + std * eps  # transform standard Gaussian to z

class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 28 * 28)
        
    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # output between [0,1]
        return x.view(-1, 1, 28, 28)  # image format reshaping (B, C, H, W)
    
    
class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar
        
        
# if __name__ == "__main__":
#     x = torch.randn(1, 1, 28, 28)
#     model = Encoder()
#     mu, logvar = model(x)
#     print(mu.shape, logvar.shape)
#     std = torch.exp(0.5*logvar)
#     std = logvar.exp()
#     eps = torch.rand_like(std)
#     print(eps)