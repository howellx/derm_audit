#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from datasets import Fitzpatrick17kDataset, ISICDataset, DDIDataset  # Replace with your datasets
from models import DeepDermClassifier  # Classifier
from diffusion_train import Diffusion, UNet  # Import diffusion and UNet classes

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_OFFSET = 50
DEFAULT_DATASET = ISICDataset
DEFAULT_CLASSIFIER = DeepDermClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="diffusion_checkpoint.pth")
    parser.add_argument("--dataset", type=str, choices=["f17k", "isic", "ddi", "from_file"], default="isic")
    parser.add_argument("--output", type=str, default="out")
    parser.add_argument("--max_images", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # Output directory
    outdir = args.output
    if not os.path.exists(outdir):
        print(f"...Creating output directory {outdir}")
        os.mkdir(outdir)

    # Dataset and Classifier
    dataset_class = DEFAULT_DATASET if args.dataset == "from_file" else ISICDataset
    classifier = DEFAULT_CLASSIFIER()
    classifier.eval().to(DEVICE)
    positive_index = classifier.positive_index

    # Load diffusion model
    diffusion = Diffusion(img_size=128, device=DEVICE)  # Match the img_size in diffusion_train.py
    model = UNet(c_in=3, c_out=3, conditional=True).to(DEVICE)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Dataset transforms
    transform = transforms.Compose([
        transforms.Resize(128),  # Match img_size from diffusion_train.py
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)  # Normalize to [-1, 1]
    ])
    dataset = dataset_class(transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Sampling loop
    for ibatch, batch in enumerate(tqdm(dataloader)):
        if ibatch * args.batch_size > args.max_images:
            break
        images, labels = batch
        images = images.to(DEVICE)

        # Generate samples for extreme conditions
        targets_min = torch.zeros(images.shape[0], device=DEVICE).float()  # Condition: min class
        targets_max = torch.ones(images.shape[0], device=DEVICE).float()  # Condition: max class

        with torch.no_grad():
            # Generate noisy images and reverse diffusion
            samples_min = diffusion.sample(model, n=images.size(0), y=targets_min)
            samples_max = diffusion.sample(model, n=images.size(0), y=targets_max)

            # Classifier predictions for original and generated images
            pred_orig = classifier(images)[:, positive_index]
            pred_min = classifier(samples_min.float() / 255.0)[:, positive_index]
            pred_max = classifier(samples_max.float() / 255.0)[:, positive_index]

        # Save images
        for i in range(images.size(0)):
            index = i + ibatch * args.batch_size
            orig_img = images[i].detach().cpu().numpy()
            min_img = samples_min[i].detach().cpu().numpy()
            max_img = samples_max[i].detach().cpu().numpy()
            orig_label = labels[i].item()
            pred_orig_ = pred_orig[i].item()
            pred_min_ = pred_min[i].item()
            pred_max_ = pred_max[i].item()

            # Combine images for saving
            combined = np.ones((128, 128 * 3 + IMG_OFFSET, 3))
            combined[:, :128, :] = orig_img.transpose(1, 2, 0)
            combined[:, 128 + IMG_OFFSET:128 * 2 + IMG_OFFSET, :] = min_img.transpose(1, 2, 0)
            combined[:, 128 * 2 + IMG_OFFSET:, :] = max_img.transpose(1, 2, 0)
            combined = (combined + 1) / 2 * 255  # De-normalize to [0, 255]
            combined = combined.astype(np.uint8)

            # Save image
            output_path = os.path.join(outdir, f"{index:05d}_{orig_label}_{pred_orig_:.3f}_{pred_min_:.3f}_{pred_max_:.3f}.png")
            Image.fromarray(combined).save(output_path)

if __name__ == "__main__":
    main()
