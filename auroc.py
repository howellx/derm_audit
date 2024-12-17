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
import csv


IMG_OFFSET = 50
NUM_CLASSES = 10
DEVICE = 'cpu' #cuda

outdir = './auroc_1000/'
if not os.path.exists(outdir):
	print(f"...Creating output directory {outdir}")
	os.mkdir(outdir)


#csv_file_path = "./out_diffusion/diffusion_results_1000.csv"
csv_file_path = "./out/results.csv"

def main():
	
	original_labels = []
	pred_orig = []
	pred_benign = []
	pred_mal = []
	mal = []
	benign = []
	
	with open(csv_file_path, mode='r') as csvfile:
		csv_reader = csv.reader(csvfile)
		next(csv_reader)

		for row in csv_reader:
			# Assuming the second column contains the original labels
			original_labels.append(int(float(row[1])))
			pred_orig.append(float(row[2]))
			pred_benign.append(float(row[3]))
			pred_mal.append(float(row[4]))
			mal.append(1)
			benign.append(0)
		
		# AUROC
		counterfactual_labels = mal + benign
		counterfactual_preds = pred_mal + pred_benign
		fpr_orig, tpr_orig, _ = roc_curve(original_labels, pred_orig)
		fpr_min, tpr_min, _ = roc_curve(counterfactual_labels, counterfactual_preds)

		roc_auc_orig = auc(fpr_orig, tpr_orig)
		roc_auc_min = auc(fpr_min, tpr_min)

		# 
		plt.figure()
		plt.plot(fpr_orig, tpr_orig, color='darkorange', lw=2, label='Prediction Original (area = {:.2f})'.format(roc_auc_orig))
		plt.plot(fpr_min, tpr_min, color='green', lw=2, label='Prediction Generated (area = {:.2f})'.format(roc_auc_min))
		plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver Operating Characteristic (AUROC)')
		plt.legend(loc="lower right")
		plt.savefig(outdir + "gan_generated_labels.png", dpi=300)
		plt.close()
		#print()    # AUROC
		# plt

if __name__ == "__main__":
    main()
