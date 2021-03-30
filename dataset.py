"""
Create dataset files (train and test) (json) from raw historical data (CSV).

Functions:

"""

# Standard Modules
import json
import os
import re

# Other Modules
import numpy as np
import pandas as pd

# Custom Modules


def process_data(data):
    # Get columns as numpy arrays and reverse so that it is from oldest to newest
    # time = np.array(data.iloc[:, 0])[::-1]
    value = data.iloc[::-1].to_numpy()

    # Mean filter (uses previous values)
    window = 10  # minutes
    overlap = 0  # fraction
    step = int(window * (1 - overlap))

    # time = np.array([time[i - 1] for i in np.arange(window, len(time), step)])
    value = np.array(
        [
            value[i - window : i, :].mean(axis=0)
            for i in np.arange(window, len(value), step)
        ]
    )

    # History and prediction intervals
    npoints = (6 + 1) * 24 * 60 // step  # 6 days of history and predict 1 day (1 week)

    # Reshape so that the shape is {batch size, sequence length, features}
    overlap = 0.95
    step = int(npoints * (1 - overlap))
    # time = time.reshape([-1, npoints])
    dataset = np.array(
        [
            value[i : i + npoints, :]
            for i in np.arange(0, len(value) - npoints + 1, step)
        ]
    )

    """
    dataset = [
        {"input": value[i, :before], "target": value[i, before:]}
        for i in range(value.shape[0])
    ]
    """

    # Normalize with respect to input vector
    """
    for i in np.arange(dataset.shape[0]):
        mean = dataset[i, :before, :].mean(axis=0)
        std = dataset[i, :before, :].std(axis=0)
        dataset[i, :, :] = (dataset[i, :, :] - mean) / std
    """

    return dataset


def split_dataset(dataset, percentage):
    # np.random.shuffle(dataset)
    n = int(len(dataset) * percentage)
    train = dataset[:n]
    test = dataset[n:]

    return train, test


def save_dataset(train, test, dataset_folder, cols):
    # Convert to dictionaries
    train = [
        {cols[j]: train[i, :, j].tolist() for j in range(len(cols))}
        for i in range(len(train))
    ]
    test = [
        {cols[j]: test[i, :, j].tolist() for j in range(len(cols))}
        for i in range(len(test))
    ]

    # Save as JSON
    with open(os.path.join(dataset_folder, "train.json"), "w") as f:
        text = json.dumps(train, indent=4)
        text = re.sub(r"(\[)\n\s*(\-?\d)", r"\1\2", text)
        text = re.sub(r"(\d,)\n\s*(\-?\d)", r"\1\2", text)
        text = re.sub(r"(\d)\n\s*(\])", r"\1\2", text)
        f.write(text)

    with open(os.path.join(dataset_folder, "test.json"), "w") as f:
        text = json.dumps(test, indent=4)
        text = re.sub(r"(\[)\n\s*(\-?\d)", r"\1\2", text)
        text = re.sub(r"(\d,)\n\s*(\-?\d)", r"\1\2", text)
        text = re.sub(r"(\d)\n\s*(\])", r"\1\2", text)
        f.write(text)


if __name__ == "__main__":
    data_folder = "data"
    dataset_folder = "dataset"
    cols = {
        # "Unix Timestamp": int,
        "Open": np.float32,
        "High": np.float32,
        "Low": np.float32,
        "Close": np.float32,
    }
    percentage = 0.95

    dataset = []

    for (dirpath, _, filenames) in os.walk(data_folder):
        for filename in filenames:
            if ".csv" in filename:
                raw_data = pd.read_csv(
                    os.path.join(dirpath, filename),
                    skiprows=1,
                    usecols=cols.keys(),
                    dtype=cols,
                )

                dataset_i = process_data(raw_data)
                dataset.append(dataset_i)

    dataset = np.vstack(dataset)

    # Split dataset into training and testing sets
    train, test = split_dataset(dataset, percentage)

    # Save train and test datasets
    save_dataset(train, test, dataset_folder, list(cols.keys()))
