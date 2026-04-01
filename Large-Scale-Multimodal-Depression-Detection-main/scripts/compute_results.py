#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""
import os
import numpy as np

def compute_mean_std(base_path, dataset, model, fusion, num_folds=10):
    # Initialize a dictionary to store lists of values for each metric
    metrics = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "Avg": [],
        "WA": [],
        "UA": [],
        "WP": [],
        "UP": [],
        "WR": [],
        "UR": [],
        "WF": [],
        "UF": []
    }

    # Loop through each fold to read the files
    for fold in range(num_folds):
        # Construct the file path
        file_path = os.path.join(base_path, f"{dataset}_{model}_{fusion}_{fold}.txt")

        # Read the single line from the file
        with open(file_path, 'r') as f:
            line = f.readline().strip()

        # Split the line into individual metric:value pairs
        parts = line.split(', ')

        # Parse each part and append the value to the corresponding metric list
        for part in parts:
            key, value = part.split(':')
            key = key.strip()  # Remove any potential whitespace
            value = float(value.strip())  # Convert string to float
            if key in metrics:
                metrics[key].append(value)
            else:
                print(f"Warning: Unknown metric {key} in file {file_path}")

    # Calculate mean and standard deviation for each metric
    results = {}
    for key in metrics:
        values = metrics[key]
        mean = np.mean(values)  # Compute mean
        std = np.std(values)    # Compute standard deviation
        # results[key] = f"{mean:.4f} ± {std:.3f}"  # Format as "mean ± std"
        results[key] = f"{mean:.4f} $\pm$ {std:.3f}"  # Format as "mean ± std"

    return results

def colored_print(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"

if __name__ == "__main__":
    # Define the parameters (modify these as needed)
    base_path = "../results"
    model = "MultiModalDepDet"

    dataset = "dvlog-dataset" # ["dvlog-dataset", "lmvd-dataset"]
    fusion = "lt_cross" # ["lt", "it", "ia", "audio", "video" ]  # Specify the fusion type you want to analyze
    num_folds = 10

    # Compute the mean and std for the specified fusion
    results = compute_mean_std(base_path, dataset, model, fusion, num_folds)

    # Define color codes
    red_color = "91"
    green_color = "92"

    # Print the results with colored prefixes and values
    # print(f"Dataset: {dataset}, Fusion: {fusion}")
    print(f"Dataset: {colored_print(dataset, red_color)}, Fusion: {colored_print(fusion, green_color)}")
    for key, value in results.items():
        if key.startswith("W"):
            colored_key_value = f"{colored_print(key, red_color)}: {colored_print(value, red_color)}"
        elif key.startswith("U"):
            colored_key_value = f"{colored_print(key, green_color)}: {colored_print(value, green_color)}"
        else:
            colored_key_value = f"{key}: {value}"
        print(colored_key_value)