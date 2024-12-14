import csv
import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from datasets import Fitzpatrick17kDataset, ISICDataset, DDIDataset
from models import Generator
from models import DeepDermClassifier 
# from models import ModelDermClassifier 
from models import ScanomaClassifier 
from models import SSCDClassifier 
from models import SIIMISICClassifier
from evaluate_classifiers import CLASSIFIER_CLASS, DATASET_CLASS

# offset the generated images from the original image by IM_OFFSET pixels
IMG_OFFSET = 50
NUM_CLASSES = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_DATASET = ISICDataset
DEFAULT_CLASSIFIER = DeepDermClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="/projectnb/ec523kb/projects/skin_tone/GAN/deepderm_isic.pth")
    parser.add_argument("--dataset", type=str, choices=["f17k", "isic", "ddi", "from_file"], default="isic")
    parser.add_argument("--classifier", type=str, choices=["deepderm", "modelderm", "scanoma", "sscd", "siimisic", "from_file"], default="deepderm")
    parser.add_argument("--output", type=str, default="out")
    parser.add_argument("--csv_file", type=str, default="./out/results.csv")
    parser.add_argument("--max_images", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    outdir = args.output
    if not os.path.exists(outdir):
        print(f"...Creating output directory {outdir}")
        os.mkdir(outdir)

    csv_file_path = args.csv_file
    with open(csv_file_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Index", "Original Label", "Prediction Original", "Prediction Min", "Prediction Max"])

    checkpoint_path = args.checkpoint_path

    if args.dataset == "from_file":
        dataset_class = DEFAULT_DATASET
    else:
        dataset_class = DATASET_CLASS[args.dataset]

    if args.classifier == "from_file":
        classifier_class = DEFAULT_CLASSIFIER
    else:
        classifier_class = CLASSIFIER_CLASS[args.classifier]

    # Load classifier model
    classifier = classifier_class()
    positive_index = classifier.positive_index
    classifier.eval()
    im_size = classifier.image_size

    # Load generator model
    generator = Generator(im_size=im_size)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    generator.load_state_dict(checkpoint['generator'])

    normalize = transforms.Normalize(mean=0.5,
                                     std=0.5)
    transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            normalize])
    dataset = dataset_class(transform=transform)
    dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            drop_last=True, 
            num_workers=2)

    generator.to(DEVICE)
    classifier.to(DEVICE)

    # for SIIM-ISIC
    try: classifier.enable_augment()
    except AttributeError: pass

    for ibatch, batch in enumerate(tqdm(dataloader)):
        if ibatch*args.batch_size > args.max_images: break
        img, label = batch 
        img = img.to(DEVICE)
        targets_min = torch.zeros(img.shape[0], dtype=torch.long).to(DEVICE)
        targets_max = (NUM_CLASSES-1)*torch.ones(img.shape[0], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            # transform images
            img_min, _ = generator(img, targets_min)
            img_max, _ = generator(img, targets_max)

            # check classifier predictions
            pred_orig = classifier(img)[:,positive_index]
            pred_min = classifier(img_min)[:,positive_index]
            pred_max = classifier(img_max)[:,positive_index]

        # save images and write to CSV
        with open(csv_file_path, mode='a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            for i_img in range(img.shape[0]):
                index = i_img + ibatch*args.batch_size
                orig = img[i_img].detach().cpu().numpy()
                min_ = img_min[i_img].detach().cpu().numpy()
                max_ = img_max[i_img].detach().cpu().numpy()
                orig_label = label[i_img]
                pred_orig_ = pred_orig[i_img].detach().cpu().numpy()
                pred_min_ = pred_min[i_img].detach().cpu().numpy()
                pred_max_ = pred_max[i_img].detach().cpu().numpy()

                full = np.ones((im_size, im_size*3+IMG_OFFSET, 3))
                full[:,:im_size,:] = orig.swapaxes(0,1).swapaxes(1,2)
                full[:,IMG_OFFSET+im_size:IMG_OFFSET+2*im_size,:] = min_.swapaxes(0,1).swapaxes(1,2)
                full[:,IMG_OFFSET+2*im_size:,:] = max_.swapaxes(0,1).swapaxes(1,2)
                full *= 0.5
                full += 0.5
                full *= 255
                full = np.require(full, dtype=np.uint8)
                out_img = Image.fromarray(full)
                out_img.save(os.path.join(outdir, 
                        "{:05d}_{:d}_{:.03f}_{:.03f}_{:.03f}.png"\
                        .format(index, 
                                int(orig_label), 
                                pred_orig_, 
                                pred_min_, 
                                pred_max_)))

                # Write to CSV
                csv_writer.writerow([index, int(orig_label), pred_orig_, pred_min_, pred_max_])

if __name__ == "__main__":
    main()

