import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
import csv
import matplotlib.pyplot as plt

original_labels = []
pred_orig = []
pred_benign = []
pred_mal = []
mal = []
benign = []

output_text_file = "./f1_score/metrics_output.txt"
csv_file_path = "./out/results.csv"

with open(csv_file_path, mode='r') as csvfile, open(output_text_file, mode='w') as outfile:
    csv_reader = csv.reader(csvfile)
    # Skip the header row
    next(csv_reader)

    for row in csv_reader:
        # Assuming the second column contains the original labels
        original_labels.append(int(row[1]))
        pred_orig.append(float(row[2]))
        pred_benign.append(float(row[3]))
        pred_mal.append(float(row[4]))
        mal.append(1)
        benign.append(0)

    # Write class distribution to file
    class_distribution = np.bincount(original_labels)
    outfile.write(f"Class Distribution: {class_distribution}\n")

    # Compute metrics for original predictions
    inverse_scores_orig = [1 - s for s in pred_orig]
    precision_orig, recall_orig, thresholds_orig = precision_recall_curve(original_labels, pred_orig, pos_label=1)

    fig, ax = plt.subplots()
    ax.plot(recall_orig, precision_orig)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.xlabel('Recall', fontsize=18)
    plt.title('Original Images')
    plt.grid(b=True, which='major', color='#999999', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig("./f1_score/original_images_plot.png", dpi=300)
    plt.close()

    numerator_orig = 2 * recall_orig * precision_orig
    denom_orig = recall_orig + precision_orig
    f1_scores_orig = np.divide(numerator_orig, denom_orig, out=np.zeros_like(denom_orig), where=(denom_orig != 0))

    best_threshold_orig = thresholds_orig[np.argmax(f1_scores_orig)]
    best_precision_orig = precision_orig[np.argmax(f1_scores_orig)]
    best_recall_orig = recall_orig[np.argmax(f1_scores_orig)]
    best_f1_score_orig = np.max(f1_scores_orig)

    outfile.write(f"Original Images:\n")
    outfile.write(f"Score Best Threshold: {best_threshold_orig}\n")
    outfile.write(f"Best Precision and Recall: {best_precision_orig}, {best_recall_orig}\n")
    outfile.write(f"Best F1-Score: {best_f1_score_orig}\n\n")

    # Compute metrics for counterfactual predictions
    pred_counterfactual = pred_benign + pred_mal
    counterfactual = benign + mal
    precision_counterfactual, recall_counterfactual, thresholds_counterfactual = precision_recall_curve(
        counterfactual, pred_counterfactual, pos_label=1
    )

    fig, ax = plt.subplots()
    ax.plot(recall_counterfactual, precision_counterfactual)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.xlabel('Recall', fontsize=18)
    plt.title('Counterfactual Images')
    plt.grid(b=True, which='major', color='#999999', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig("./f1_score/counterfactual_images_plot.png", dpi=300)
    plt.close()

    numerator_counterfactual = 2 * recall_counterfactual * precision_counterfactual
    denom_counterfactual = recall_counterfactual + precision_counterfactual
    f1_scores_counterfactual = np.divide(
        numerator_counterfactual, denom_counterfactual, out=np.zeros_like(denom_counterfactual), where=(denom_counterfactual != 0)
    )

    best_threshold_counterfactual = thresholds_counterfactual[np.argmax(f1_scores_counterfactual)]
    best_precision_counterfactual = precision_counterfactual[np.argmax(f1_scores_counterfactual)]
    best_recall_counterfactual = recall_counterfactual[np.argmax(f1_scores_counterfactual)]
    best_f1_score_counterfactual = np.max(f1_scores_counterfactual)

    outfile.write(f"Counterfactual Images:\n")
    outfile.write(f"Score Best Threshold: {best_threshold_counterfactual}\n")
    outfile.write(f"Best Precision and Recall: {best_precision_counterfactual}, {best_recall_counterfactual}\n")
    outfile.write(f"Best F1-Score: {best_f1_score_counterfactual}\n\n")


    # Compute metrics for combined predictions
    pred_combined = pred_orig + pred_benign + pred_mal
    combined = original_labels + benign + mal
    precision_combined, recall_combined, thresholds_combined = precision_recall_curve(
        combined, pred_combined, pos_label=1
    )

    fig, ax = plt.subplots()
    ax.plot(recall_combined, precision_combined)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.xlabel('Recall', fontsize=18)
    plt.title('combined Images')
    plt.grid(b=True, which='major', color='#999999', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig("./f1_score/combined_images_plot.png", dpi=300)
    plt.close()

    numerator_combined = 2 * recall_combined * precision_combined
    denom_combined = recall_combined + precision_combined
    f1_scores_combined = np.divide(
        numerator_combined, denom_combined, out=np.zeros_like(denom_combined), where=(denom_combined != 0)
    )

    best_threshold_combined = thresholds_combined[np.argmax(f1_scores_combined)]
    best_precision_combined = precision_combined[np.argmax(f1_scores_combined)]
    best_recall_combined = recall_combined[np.argmax(f1_scores_combined)]
    best_f1_score_combined = np.max(f1_scores_combined)

    outfile.write(f"Combined Images:\n")
    outfile.write(f"Score Best Threshold: {best_threshold_combined}\n")
    outfile.write(f"Best Precision and Recall: {best_precision_combined}, {best_recall_combined}\n")
    outfile.write(f"Best F1-Score: {best_f1_score_combined}\n")
