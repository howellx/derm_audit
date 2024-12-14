import pandas as pd

def count_mel_labels(csv_file_path):
    """
    Counts the number of entries with MEL = 0 and MEL = 1 in the given CSV file.

    Args:
        csv_file_path (str): Path to the ISIC 2019 Ground Truth CSV file.

    Returns:
        dict: A dictionary with counts for MEL = 0 and MEL = 1.
    """
    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Count the occurrences of MEL == 0 and MEL == 1
    mel_label_counts = df['MEL'].value_counts()

    # Format the result as a dictionary
    result = {
        'MEL = 0': mel_label_counts.get(0.0, 0),
        'MEL = 1': mel_label_counts.get(1.0, 0)
    }

    return result

# Example usage
if __name__ == "__main__":
    csv_file_path = "./data/isic/ISIC_2019_Training_GroundTruth.csv"  # Replace with the actual path
    counts = count_mel_labels(csv_file_path)
    print(f"Count of MEL labels: {counts}")

