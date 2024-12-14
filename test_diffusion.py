#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from datasets import ISICDataset  # Replace with your dataset
from models import DeepDermClassifier  # Replace with your classifier
from diffusion import DDPM, ContextUnet  # Correctly import DDPM and ContextUnet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_OFFSET = 10  # Reduced offset for tighter spacing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="diffusion_checkpoint.pth")
    parser.add_argument("--output", type=str, default="out")
    parser.add_argument("--max_images", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--guide_w", type=float, default=2.0)
    args = parser.parse_args()

    outdir = args.output
    if not os.path.exists(outdir):
        print(f"...Creating output directory {outdir}")
        os.mkdir(outdir)

    # Setup data loading
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = ISICDataset(transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        drop_last=True
    )

    # Initialize and load model
    ddpm = DDPM(
        nn_model=ContextUnet(in_channels=3, n_feat=128, n_classes=2),
        betas=(1e-4, 0.02),
        n_T=400,
        device=DEVICE
    ).to(DEVICE)

    checkpoint = torch.load(args.checkpoint_path, map_location=DEVICE)
    ddpm.load_state_dict(checkpoint['model'])
    ddpm.eval()

    # Sampling loop
    for ibatch, batch in enumerate(tqdm(dataloader)):
        if ibatch * args.batch_size >= args.max_images:
            break
            
        images, labels = batch
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        samples = []
        with torch.no_grad():
            # Generate melanoma samples
            x_gen_mel, _ = ddpm.sample(
                n_sample=args.batch_size,
                size=(3, 224, 224),
                device=DEVICE,
                guide_w=5.0,
                condition=torch.ones(args.batch_size, dtype=torch.long, device=DEVICE),
                x_init=images,
                denoise_strength=0.9
            )
            samples.append(x_gen_mel)

            # Generate benign samples
            x_gen_benign, _ = ddpm.sample(
                n_sample=args.batch_size,
                size=(3, 224, 224),
                device=DEVICE,
                guide_w=3.0,
                condition=torch.zeros(args.batch_size, dtype=torch.long, device=DEVICE),
                x_init=images,
                denoise_strength=0.8
            )
            samples.append(x_gen_benign)

        # Save images
        for i in range(images.size(0)):
            index = i + ibatch * args.batch_size
            orig_img = images[i].cpu().numpy().transpose(1, 2, 0)
            orig_label = labels[i].item()

            # Create combined image (original + melanoma + benign)
            combined_width = 224 * 3 + IMG_OFFSET * 2
            combined = np.ones((224, combined_width, 3))
            
            # Add original image
            combined[:, :224, :] = orig_img
            
            # Add generated samples
            for j, gen_batch in enumerate(samples):
                x_start = 224 * (j + 1) + IMG_OFFSET * (j + 1)
                gen_img = gen_batch[i].cpu().numpy().transpose(1, 2, 0)
                combined[:, x_start:x_start+224, :] = gen_img

            # Denormalize
            combined = (combined + 1) / 2 * 255
            combined = np.clip(combined, 0, 255).astype(np.uint8)
            
            # Save
            output_path = os.path.join(outdir, f"{index:05d}_orig{int(orig_label)}_mel_benign.png")
            Image.fromarray(combined).save(output_path)

if __name__ == "__main__":
    main()

