#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np
import csv
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from datasets import ISICDataset  # Replace with your dataset
from models import DeepDermClassifier  # Replace with your classifier
from diffusion import DDPM, ContextUnet  # Correctly import DDPM and ContextUnet
from evaluate_classifiers import CLASSIFIER_CLASS, DATASET_CLASS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_OFFSET = 10  # Spacing between images
DEFAULT_CLASSIFIER = DeepDermClassifier

# Function to denormalize images
def denormalize(image_tensor):
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)  # Move to HxWxC
    image = (image * 0.5) + 0.5  # Reverse normalization [-1, 1] -> [0, 1]
    image = np.clip(image * 255, 0, 255).astype(np.uint8)  # Scale to [0, 255]
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="diffusion_checkpoint.pth")
    parser.add_argument("--output", type=str, default="out")
    parser.add_argument("--max_images", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--guide_w", type=float, default=2.0)
    parser.add_argument("--classifier", type=str, choices=["deepderm", "modelderm", "scanoma", "sscd", "siimisic", "from_file"], default="deepderm")
    args = parser.parse_args()

    outdir = args.output
    melanoma_dir = os.path.join(outdir, "melanoma")
    benign_dir = os.path.join(outdir, "benign")
    csv_path = os.path.join(outdir, "results.csv")

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(melanoma_dir, exist_ok=True)
    os.makedirs(benign_dir, exist_ok=True)

    with open(csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Index", "Ground Truth", "Original Prediction", "Benign Prediction", "Melanoma Prediction"])

    if args.classifier == "from_file":
        classifier_class = DEFAULT_CLASSIFIER
    else:
        classifier_class = CLASSIFIER_CLASS[args.classifier]

    classifier = classifier_class()
    positive_index = classifier.positive_index
    classifier.eval()

    im_size = 256
    transform = transforms.Compose([
        transforms.Resize((im_size, im_size)),
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

    ddpm = DDPM(
        nn_model=ContextUnet(in_channels=3, n_feat=128, n_classes=2),
        betas=(1e-4, 0.02),
        n_T=400,
        device=DEVICE
    ).to(DEVICE)

    checkpoint = torch.load(args.checkpoint_path, map_location=DEVICE)
    ddpm.load_state_dict(checkpoint['model'])
    ddpm.eval()

    classifier.to(DEVICE)

    try: 
        classifier.enable_augment()
    except AttributeError: 
        pass

    for ibatch, batch in enumerate(tqdm(dataloader)):
        if ibatch * args.batch_size >= args.max_images:
            break
            
        images, labels = batch  # No filenames, just images and labels
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            x_gen_mel, _ = ddpm.sample(
                n_sample=args.batch_size,
                size=(3, im_size, im_size),
                device=DEVICE,
                guide_w=10.0,
                condition=torch.ones(args.batch_size, dtype=torch.long, device=DEVICE),
                x_init=images,
                denoise_strength=0.8
            )

            x_gen_benign, _ = ddpm.sample(
                n_sample=args.batch_size,
                size=(3, im_size, im_size),
                device=DEVICE,
                guide_w=10.0,
                condition=torch.zeros(args.batch_size, dtype=torch.long, device=DEVICE),
                x_init=images,
                denoise_strength=0.8
            )

            pred_orig = classifier(images)[:, positive_index]
            pred_mel = classifier(x_gen_mel)[:, positive_index]
            pred_ben = classifier(x_gen_benign)[:, positive_index]

        for i in range(images.size(0)):
            index = i + ibatch * args.batch_size
            orig_label = labels[i].item()

            pred_orig_ = pred_orig[i].detach().cpu().numpy()
            pred_mel_ = pred_mel[i].detach().cpu().numpy()
            pred_ben_ = pred_ben[i].detach().cpu().numpy()

            # Denormalize images
            orig_img = denormalize(images[i])
            mel_img = denormalize(x_gen_mel[i])
            ben_img = denormalize(x_gen_benign[i])

            # Save melanoma image
            Image.fromarray(mel_img).save(os.path.join(melanoma_dir, f"melanoma_{index:05d}.png"))

            # Save benign image
            Image.fromarray(ben_img).save(os.path.join(benign_dir, f"benign_{index:05d}.png"))

            # Create combined image
            combined_width = im_size * 3 + IMG_OFFSET * 2
            combined = np.ones((im_size, combined_width, 3), dtype=np.uint8)

            # Add original image
            combined[:, :im_size, :] = orig_img
            
            # Add benign sample
            x_start = im_size  + IMG_OFFSET 
            combined[:, x_start:x_start+im_size, :] = ben_img

            # Add melanoma sample
            x_start = im_size* 2 + IMG_OFFSET * 2
            combined[:, x_start:x_start+im_size, :] = mel_img


            # Save combined image
            Image.fromarray(combined).save(os.path.join(
                outdir, 
                f"combined_{index:05d}_{orig_label}_{pred_orig_:.3f}_{pred_ben_:.3f}_{pred_mel_:.3f}.png"
            ))

            # Append predictions to CSV
            with open(csv_path, mode='a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([index, orig_label, pred_orig_, pred_ben_, pred_mel_])


if __name__ == "__main__":
    main()
