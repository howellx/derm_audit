from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import ISICDataset  # Replace with your dataset
import torch.optim as optim
import csv
import os

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
        super().__init__()
        self.same_channels = (in_channels == out_channels)
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

def convert_to_one_hot(labels, num_classes=2):
    """Convert binary labels to one-hot encoding"""
    labels = labels.long()
    one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=128, n_classes=2):
        super().__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        # Using AdaptiveAvgPool2d to get to 8x8
        self.to_vec = nn.Sequential(
            nn.AdaptiveAvgPool2d((8,8)),
            nn.GELU()
        )

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        # From 8x8 back to 64x64
        # (8 - 1)*8 + 8 = 56 + 8 =64
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, kernel_size=8, stride=8),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        x = self.init_conv(x)
        down1 = self.down1(x)   # 128x128
        down2 = self.down2(down1)  # 64x64
        hiddenvec = self.to_vec(down2)  # 8x8

        c_one_hot = convert_to_one_hot(c, self.n_classes)
        
        context_mask = context_mask.view(-1, 1)
        context_mask = context_mask.repeat(1, self.n_classes)
        context_mask = (-1*(1-context_mask))
        c_one_hot = c_one_hot * context_mask

        cemb1 = self.contextembed1(c_one_hot).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c_one_hot).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)  # 64x64
        up2 = self.up1(cemb1*up1 + temb1, down2)  # from 64x64 to 128x128
        up3 = self.up2(cemb2*up2 + temb2, down1)  # from 128x128 to 256x256
        out = self.out(torch.cat((up3, x), 1))    # final 256x256
        return out

def ddpm_schedules(beta1, beta2, T):
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super().__init__()
        self.nn_model = nn_model.to(device)
        
        # Register DDPM schedule parameters
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)
            
        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)

        # Create noisy image
        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )

        c = (c > 0.5).long()  
        context_mask = torch.bernoulli(torch.zeros_like(c.float()) + self.drop_prob).to(self.device)

        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w=0.0, condition=None, x_init=None, denoise_strength=0.7):
        # Sampling logic (unchanged)
        if x_init is not None:
            start_t = int(self.n_T * denoise_strength)
            noise = torch.randn_like(x_init)
            x_i = self.sqrtab[start_t] * x_init + self.sqrtmab[start_t] * noise
            start_step = start_t
        else:
            x_i = torch.randn(n_sample, *size).to(device)
            start_step = self.n_T

        if condition is not None:
            c_i = condition
        else:
            c_i = torch.arange(0, 2).to(device)
            c_i = c_i.repeat(int(n_sample/c_i.shape[0]))

        context_mask = torch.zeros_like(c_i).to(device)
        x_i_store = []

        for i in range(start_step, 0, -1):
            print(f'sampling timestep {i}', end='\r')
            t_is = torch.tensor([i / self.n_T], device=device).reshape(1, 1, 1, 1)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            x_i_double = x_i.repeat(2, 1, 1, 1)
            t_is_double = t_is.repeat(2, 1, 1, 1)
            c_i_double = c_i.repeat(2)
            context_mask_double = context_mask.repeat(2)
            context_mask_double[n_sample:] = 1.

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.nn_model(x_i_double, c_i_double, t_is_double, context_mask_double)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2

            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())

        return x_i, x_i_store

def train_isic():
    # Training parameters
    n_epoch = 30
    batch_size = 8
    n_T = 400
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_classes = 2
    n_feat = 128
    image_size = 256  # Use a clean power-of-two image size
    in_channels = 3
    lrate = 1e-4
    save_interval = 5
    save_path = "diffusion_checkpoint.pth"

    ddpm = DDPM(
        nn_model=ContextUnet(in_channels=in_channels, n_feat=n_feat, n_classes=n_classes),
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        drop_prob=0.1
    )
    ddpm.to(device)

    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = ISICDataset(transform=tf)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    csv_file = "training_loss.csv"
    file_exists = os.path.isfile(csv_file)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # Linear LR decay
        optim.param_groups[0]['lr'] = lrate * (1 - ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None

        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            
            try:
                loss = ddpm(x, c)
                loss.backward()
                
                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
                
                pbar.set_description(f"loss: {loss_ema:.4f}")
                optim.step()
                
            except Exception as e:
                print(f"Error in batch: {str(e)}")
                continue

        # Save model checkpoint
        torch.save({
            'model': ddpm.state_dict(),
            'optimizer': optim.state_dict(),
            'epoch': ep
        }, save_path)
        print(f"Saved checkpoint for epoch {ep}")

        # Write loss to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists and ep == 0:
                # Write header if file did not exist before
                writer.writerow(["epoch", "loss_ema"])
            writer.writerow([ep, loss_ema])

        # Backup checkpoint every save_interval epochs
        if ep % save_interval == 0:
            backup_path = f"{save_path}.{ep}"
            torch.save({
                'model': ddpm.state_dict(),
                'optimizer': optim.state_dict(),
                'epoch': ep
            }, backup_path)
            print(f"Saved backup checkpoint to {backup_path}")


if __name__ == "__main__":
    train_isic()
