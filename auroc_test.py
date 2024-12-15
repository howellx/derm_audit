import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from datasets import ISICDataset
from models import Generator
from models import DeepDermClassifier


IMG_OFFSET = 50
NUM_CLASSES = 10
DEVICE = 'cpu' #cuda


def main():
    #
    checkpoint_path = "checkpoint.pth"
    output_dir = "out"
    max_images = 1000
    batch_size = 4


    if not os.path.exists(output_dir):
        print(f"...Creating output directory {output_dir}")
        os.mkdir(output_dir)


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    dataset = ISICDataset(transform=transform)
    total_samples = len(dataset)


    sample_size = min(max(1, int(total_samples * 0.05)), max_images)
    indices = np.random.choice(total_samples, sample_size, replace=False)
    subset = Subset(dataset, indices)

    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2
    )

    # 加载生成器
    generator = Generator(im_size=224)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    generator.load_state_dict(checkpoint['generator'])
    generator.to(DEVICE)
    generator.eval()

    # 加载分类器
    classifier = DeepDermClassifier()
    positive_index = classifier.positive_index
    classifier.to(DEVICE)
    classifier.eval()

    all_labels = []
    all_preds_orig = []
    all_preds_min = []
    all_preds_max = []


    for ibatch, batch in enumerate(tqdm(dataloader)):
        if ibatch * batch_size >= max_images:
            break
        img, label = batch
        img = img.to(DEVICE)

        targets_min = torch.zeros(img.shape[0], dtype=torch.long).to(DEVICE)
        targets_max = (NUM_CLASSES - 1) * torch.ones(img.shape[0], dtype=torch.long).to(DEVICE)

        with torch.no_grad():

            img_min, _ = generator(img, targets_min)
            img_max, _ = generator(img, targets_max)


            pred_orig = classifier(img)[:, positive_index]
            pred_min = classifier(img_min)[:, positive_index]
            pred_max = classifier(img_max)[:, positive_index]


        all_labels.extend(label.cpu().numpy())
        all_preds_orig.extend(pred_orig.cpu().numpy())
        all_preds_min.extend(pred_min.cpu().numpy())
        all_preds_max.extend(pred_max.cpu().numpy())

        '''
        for i_img in range(img.shape[0]):
            index = i_img + ibatch * batch_size
            orig = img[i_img].detach().cpu().numpy()
            min_ = img_min[i_img].detach().cpu().numpy()
            max_ = img_max[i_img].detach().cpu().numpy()
            orig_label = label[i_img]
            pred_orig_ = pred_orig[i_img].item()
            pred_min_ = pred_min[i_img].item()
            pred_max_ = pred_max[i_img].item()

            # 
            full = np.ones((224, 224 * 3 + IMG_OFFSET, 3))
            full[:, :224, :] = orig.swapaxes(0, 1).swapaxes(1, 2)
            full[:, IMG_OFFSET + 224:IMG_OFFSET + 2 * 224, :] = min_.swapaxes(0, 1).swapaxes(1, 2)
            full[:, IMG_OFFSET + 2 * 224:, :] = max_.swapaxes(0, 1).swapaxes(1, 2)
            full *= 0.5
            full += 0.5
            full *= 255
            full = np.require(full, dtype=np.uint8)

            out_img = Image.fromarray(full)
            out_img.save(os.path.join(
                output_dir,
                "{:05d}_{:d}_{:.03f}_{:.03f}_{:.03f}.png".format(
                    index,
                    int(orig_label),
                    pred_orig_,
                    pred_min_,
                    pred_max_
                )
            ))'''

    # AUROC
    fpr_orig, tpr_orig, _ = roc_curve(all_labels, all_preds_orig)
    fpr_min, tpr_min, _ = roc_curve(all_labels, all_preds_min)
    fpr_max, tpr_max, _ = roc_curve(all_labels, all_preds_max)

    roc_auc_orig = auc(fpr_orig, tpr_orig)
    roc_auc_min = auc(fpr_min, tpr_min)
    roc_auc_max = auc(fpr_max, tpr_max)

    # 绘制 AUROC 曲线
    plt.figure()
    plt.plot(fpr_orig, tpr_orig, color='darkorange', lw=2, label='Prediction Original (area = {:.2f})'.format(roc_auc_orig))
    plt.plot(fpr_min, tpr_min, color='green', lw=2, label='Min Generated (area = {:.2f})'.format(roc_auc_min))
    plt.plot(fpr_max, tpr_max, color='blue', lw=2, label='Max Generated (area = {:.2f})'.format(roc_auc_max))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (AUROC)')
    plt.legend(loc="lower right")
    plt.show()
    #print()    # AUROC
    fpr_orig, tpr_orig, _ = roc_curve(all_labels, all_preds_orig)
    fpr_min, tpr_min, _ = roc_curve(all_labels, all_preds_min)
    fpr_max, tpr_max, _ = roc_curve(all_labels, all_preds_max)

    roc_auc_orig = auc(fpr_orig, tpr_orig)
    roc_auc_min = auc(fpr_min, tpr_min)
    roc_auc_max = auc(fpr_max, tpr_max)
    print("False Positive Rate (FPR) for original images:", fpr_orig)
    print("True Positive Rate (TPR) for original images:", tpr_orig)
    # plt

if __name__ == "__main__":
    main()


