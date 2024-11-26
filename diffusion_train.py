import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import ISICDataset  # Assuming ISICDataset is available

class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, emb_dim=256, n_classes=10):
        """
        c_in: Number of image channels in input.
        c_out: Number of image channels in output.
        emb_dim: Length of conditional embedding vector.
        n_classes: Number of classifier output classes.
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.inc = Block(c_in, 64, emb_dim)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 256)
        self.bot = Block(256, 512, emb_dim)
        self.up1 = Up(512, 128)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        # Class embedding for DeepDerm classifier labels
        self.class_embed = nn.Embedding(n_classes, emb_dim)

    def temporal_encoding(self, t):
        """Sinusoidal temporal encoding."""
        half_dim = self.emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

    def forward(self, x, t, y):
        """
        x: input image.
        t: diffusion timestep.
        y: target classifier labels for conditional embedding.
        """
        # Generate temporal and class embeddings
        t_emb = self.temporal_encoding(t)
        y_emb = self.class_embed(y)
        c = t_emb + y_emb  # Combine embeddings

        # UNet forward pass
        x1 = self.inc(x, c)
        x2 = self.down1(x1, c)
        x3 = self.down2(x2, c)
        x4 = self.down3(x3, c)
        x_bot = self.bot(x4, c)
        x = self.up1(x_bot, x3, c)
        x = self.up2(x, x2, c)
        x = self.up3(x, x1, c)
        return self.outc(x)

from models import DeepDermClassifier  # Import DeepDermClassifier

def train_conditional_diffusion():
    # Hyperparameters
    n_epochs = 500
    batch_size = 4
    accumulate_steps = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    dataset = ISICDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Models
    diffusion_model = UNet(c_in=3, c_out=3, emb_dim=256, n_classes=10).to(device)
    classifier = DeepDermClassifier().to(device)
    classifier.eval()  # Freeze classifier during training

    # Optimizer and Loss
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=2e-4, betas=(0.0, 0.9))
    mse_loss = nn.MSELoss()

    # Noise schedule (beta values)
    timesteps = 1000
    beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alpha = 1 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)

    for epoch in range(n_epochs):
        for batch_idx, (real_images, labels) in enumerate(tqdm(dataloader)):
            real_images = real_images.to(device)
            labels = labels.to(device)  # Target labels for conditioning
            bsz = real_images.size(0)

            # Random timesteps
            t = torch.randint(0, timesteps, (bsz,), device=device).long()

            # Add noise to images
            noise = torch.randn_like(real_images).to(device)
            alpha_t = alpha_cumprod[t][:, None, None, None]
            noisy_images = torch.sqrt(alpha_t) * real_images + torch.sqrt(1 - alpha_t) * noise

            # Predict noise with conditional UNet
            pred_noise = diffusion_model(noisy_images, t.float(), labels)

            # Compute loss
            loss = mse_loss(pred_noise, noise) / accumulate_steps
            loss.backward()

            if (batch_idx + 1) % accumulate_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")

# Train the diffusion model with classifier labels
train_conditional_diffusion()

