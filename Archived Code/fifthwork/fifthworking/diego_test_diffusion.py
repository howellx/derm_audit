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
from evaluate_classifiers import CLASSIFIER_CLASS, DATASET_CLASS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_OFFSET = 10  # Spacing between images
DEFAULT_CLASSIFIER = DeepDermClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="diffusion_checkpoint.pth")
    parser.add_argument("--output", type=str, default="out")
    parser.add_argument("--max_images", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--guide_w", type=float, default=2.0)
    parser.add_argument("--classifier", type=str, choices=["deepderm", "modelderm", "scanoma", "sscd", "siimisic", "from_file"], default="deepderm")
    args = parser.parse_args()

    outdir = args.output
    if not os.path.exists(outdir):
        print(f"...Creating output directory {outdir}")
        os.mkdir(outdir)

    if args.classifier == "from_file":
        classifier_class = DEFAULT_CLASSIFIER
    else:
        classifier_class = CLASSIFIER_CLASS[args.classifier]

    classifier = classifier_class()
    positive_index = classifier.positive_index
    classifier.eval()

    # Use a fixed image size of 256x256
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

    classifier.to(DEVICE)

    try: 
        classifier.enable_augment()
    except AttributeError: 
        pass

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
                size=(3, im_size, im_size),
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
                size=(3, im_size, im_size),
                device=DEVICE,
                guide_w=5.0,
                condition=torch.zeros(args.batch_size, dtype=torch.long, device=DEVICE),
                x_init=images,
                denoise_strength=0.8
            )
            samples.append(x_gen_benign)

            pred_orig = classifier(images)[:, positive_index]
            pred_mel = classifier(x_gen_mel)[:, positive_index]
            pred_ben = classifier(x_gen_benign)[:, positive_index]

        # Save images
        for i in range(images.size(0)):
            index = i + ibatch * args.batch_size
            orig_img = images[i].cpu().numpy().transpose(1, 2, 0)
            orig_label = labels[i].item()

            pred_orig_ = pred_orig[i].detach().cpu().numpy()
            pred_mel_ = pred_mel[i].detach().cpu().numpy()
            pred_ben_ = pred_ben[i].detach().cpu().numpy()

            # We have 3 images: original, melanoma, benign
            # Total width: 3 * im_size + 2 * IMG_OFFSET
            combined_width = im_size * 3 + IMG_OFFSET * 2
            combined = np.ones((im_size, combined_width, 3), dtype=np.float32)

            # Add original image at position 0
            combined[:, :im_size, :] = orig_img

            # Add generated melanoma sample
            x_start = im_size + IMG_OFFSET
            gen_img_mel = x_gen_mel[i].cpu().numpy().transpose(1, 2, 0)
            combined[:, x_start:x_start+im_size, :] = gen_img_mel

            # Add generated benign sample
            x_start = im_size * 2 + IMG_OFFSET * 2
            gen_img_ben = x_gen_benign[i].cpu().numpy().transpose(1, 2, 0)
            combined[:, x_start:x_start+im_size, :] = gen_img_ben

            # Denormalize
            combined = (combined + 1) / 2 * 255
            combined = np.clip(combined, 0, 255).astype(np.uint8)

            out_img = Image.fromarray(combined)
            out_img.save(os.path.join(
                outdir, 
                "{:05d}_{:d}_{:.03f}_{:.03f}_{:.03f}.png".format(
                    index, 
                    int(orig_label), 
                    pred_orig_, 
                    pred_mel_, 
                    pred_ben_
                )
            ))


if __name__ == "__main__":
    main()

