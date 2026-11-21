import torch
import torch.nn.functional as F
from torch import optim

import matplotlib.pyplot as plt
import numpy as np

from get_data import load_data
from vae_model import VAE, reparameterise


def loss_function(recon_x, x, mu, logvar):
    # Keep values in valid ranges
    x = torch.clamp(x, 0, 1)
    recon_x = torch.clamp(recon_x, 1e-6, 1 - 1e-6)

    # Reconstruction Loss
    recon_loss = F.binary_cross_entropy(
        recon_x,
        x,
        reduction='sum'
    )

    # KL Divergence
    kl_div = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    # Total VAE Loss
    return recon_loss + kl_div


def visualise_mnist(dataset, num_images=10):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 2))
    for i in range(num_images):
        image, label = dataset[i]
        axes[i].imshow(image.squeeze(), cmap="gray")
        axes[i].axis("off")
    plt.show()
    
    
def train(
    vae,
    train_loader,
    optimizer,
    device="cpu",
    epochs=1
):
    vae.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.4f}")
        
def generate_images(vae, device="cpu", num_samples=10):
    with torch.no_grad():
        z = torch.randn(num_samples, 2).to(device)  # sampling from N(0,1)
        samples = vae.decoder(z).cpu()
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 2))
        for i in range(num_samples):
            axes[i].imshow(samples[i].squeeze(), cmap="gray")
            axes[i].axis("off")
        plt.show()
        

def visualise_latent_space(vae, test_loader, device="cpu"):
    vae.eval()
    latents, labels = [], []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            mu, logvar = vae.encoder(data)
            z = reparameterise(mu, logvar)
            latents.append(z.cpu().numpy())
            labels.append(label.numpy())
    latents = np.concatenate(latents)
    labels = np.concatenate(labels)
    plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', alpha=0.5)
    plt.colorbar()
    plt.show()
    
def interpolate_images(vae, test_loader, device="cpu"):
    vae.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        img1, img2 = data[2].to(device), data[12].to(device)
        mu1, logvar1 = vae.encoder(img1.unsqueeze(0))
        mu2, logvar2 = vae.encoder(img2.unsqueeze(0))
        z1, z2 = reparameterise(mu1, logvar1), reparameterise(mu2, logvar2)
        alphas = np.linspace(0, 1, 10)
        interpolated_images = [vae.decoder((1 - a) * z1 + a * z2).cpu() for a in alphas]
        fig, axes = plt.subplots(1, 10, figsize=(15, 2))
        for i in range(10):
            axes[i].imshow(interpolated_images[i].squeeze(), cmap="gray")
            axes[i].axis("off")
        plt.show()

def main():
    train_dataset, test_dataset, train_loader, test_loader = load_data()
    # visualise_mnist(train_dataset)
    
    vae = VAE(latent_dim=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    
    train(
        vae,
        train_loader,
        optimizer,
        device
    )
    
    generate_images(vae)
    visualise_latent_space(vae, test_loader, device)
    interpolate_images(vae, test_loader, device)
    
    
    
if __name__ == "__main__":
    main()