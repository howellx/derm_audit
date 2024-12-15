import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from scipy import linalg
from tqdm import tqdm
from torch.utils.data import DataLoader
import gc
from torchvision import transforms

class InceptionV3Feature(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.model = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3, inception.maxpool1,
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            inception.maxpool2, inception.Mixed_5b,
            inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b,
            inception.Mixed_6c, inception.Mixed_6d,
            inception.Mixed_6e, inception.Mixed_7a,
            inception.Mixed_7b, inception.Mixed_7c,
            inception.avgpool
        )
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x):
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        features = self.model(x)
        return features.squeeze()

def calculate_activation_statistics(feature_extractor, dataloader, device):
    features = []
    batch_size = 8  # Reduced batch size
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            
            # Process in smaller sub-batches if needed
            for i in range(0, len(batch), batch_size):
                sub_batch = batch[i:i+batch_size].to(device)
                batch_features = feature_extractor(sub_batch).cpu().numpy()
                features.append(batch_features)
                
                # Clear cache
                torch.cuda.empty_cache()
                del sub_batch
    
    features = np.concatenate(features, axis=0)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    
    return mu, sigma

def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return (diff.dot(diff) + 
            np.trace(sigma1) + 
            np.trace(sigma2) - 
            2 * tr_covmean)

def compute_fid_score(ddpm, real_dataloader, n_samples, image_size, device):
    # Smaller batch sizes
    gen_batch_size = 4  # Reduced from 32
    eval_batch_size = 8  # For feature extraction
    
    feature_extractor = InceptionV3Feature().to(device)
    
    # Calculate real image statistics
    real_mu, real_sigma = calculate_activation_statistics(
        feature_extractor, real_dataloader, device
    )
    
    # Generate images in smaller batches
    generated_images = []
    n_batches = n_samples // gen_batch_size + (1 if n_samples % gen_batch_size != 0 else 0)
    
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Generating images"):
            if len(generated_images) * gen_batch_size >= n_samples:
                break
                
            curr_batch_size = min(gen_batch_size, n_samples - len(generated_images) * gen_batch_size)
            
            try:
                fake_images, _ = ddpm.sample(
                    n_sample=curr_batch_size,
                    size=(3, image_size, image_size),
                    device=device,
                    guide_w=2.0
                )
                generated_images.append(fake_images.cpu())  # Move to CPU immediately
                
                # Clear cache
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                print(f"Error generating batch {i}: {e}")
                continue
    
    # Concatenate generated images and create dataloader
    generated_images = torch.cat(generated_images, dim=0)[:n_samples]
    fake_dataloader = DataLoader(
        generated_images,
        batch_size=eval_batch_size,
        shuffle=False
    )
    
    # Calculate generated image statistics
    fake_mu, fake_sigma = calculate_activation_statistics(
        feature_extractor, fake_dataloader, device
    )
    
    # Clean up
    del generated_images
    del feature_extractor
    torch.cuda.empty_cache()
    gc.collect()
    
    # Calculate FID score
    return calculate_fid(real_mu, real_sigma, fake_mu, fake_sigma)

def evaluate_model(ddpm, dataset, n_samples=1000, image_size=256):
    device = next(ddpm.parameters()).device
    
    # Use smaller batch size for real images
    real_dataloader = DataLoader(
        dataset,
        batch_size=8,  # Reduced batch size
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    
    # Calculate FID score
    fid_score = compute_fid_score(
        ddpm=ddpm,
        real_dataloader=real_dataloader,
        n_samples=n_samples,
        image_size=image_size,
        device=device
    )
    
    return fid_score

if __name__ == "__main__":
    from diffusion import DDPM, ContextUnet
    from datasets import ISICDataset
    import torch
    
    # Setup dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = ISICDataset(transform=transform)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddpm = DDPM(
        nn_model=ContextUnet(in_channels=3, n_feat=128, n_classes=2),
        betas=(1e-4, 0.02),
        n_T=400,
        device=device,
        drop_prob=0.1
    ).to(device)
    
    # Load weights
    checkpoint = torch.load("diffusion_checkpoint.pth")
    ddpm.load_state_dict(checkpoint['model'])
    ddpm.eval()
    
    # Calculate FID score with smaller batches
    fid_score = evaluate_model(
        ddpm=ddpm,
        dataset=dataset,
        n_samples=8,
        image_size=256
    )
    
    print(f"FID Score: {fid_score}")
