import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

from vae.classifier import VAEClassifier
from vae.utils.logger import logger
from vae.utils.utils import get_dataset_path

import json
import logging
import argparse

# Set the level for this module
logger.setLevel(logging.INFO)   # or DEBUG, WARNING, ERROR, CRITICAL

# CONFIGURATION
# Check for GPU availability
TR_SIZE = [2, 3, 4, 5, 6, 8, 10, 16]  # training size n


def shuffle_split(X, y, tr_sz=10, random_state=42):
    """
    :param X, y: data to split
    :param tr_sz: training size
    :param random_state:
    :return: dict(encoder=train_2Darray, labels=labels),
             dict(encoder=test_2Darray, labels=labels)
    """
    X_shuffled, y_shuffled = shuffle(X, y, random_state=random_state)
    labels = np.array(y_shuffled, dtype=bool)
    train_idx = np.hstack((np.where(labels)[0][:tr_sz], np.where(~labels)[0][:tr_sz]))
    test_idx = np.hstack((np.where(labels)[0][tr_sz:], np.where(~labels)[0][tr_sz:]))

    logger.debug("Splitting X and labels.")
    labels = np.squeeze(labels)
    tr_encoder = X_shuffled[train_idx]
    te_encoder = X_shuffled[test_idx]

    return dict(encoder=tr_encoder, labels=labels[train_idx]), \
           dict(encoder=te_encoder, labels=labels[test_idx])


def classify_data(tr_size, X, labels, random_state=42):
    """
    :param tr_size: training size n
    :param X: data to classify
    :param labels: labels
    :param random_state:
    :return: predictions and scores in [0,1]
    """
    data_train, data_test = shuffle_split(X, labels, tr_sz=tr_size, random_state=random_state)

    clf = VAEClassifier()
    clf.fit(data_train['encoder'], data_train['labels'])

    y_pred = clf.predict(data_test['encoder'])
    predVI = clf.predict_proba(data_test['encoder'])
    
    logger.info(f"Test accuracy: {round(accuracy_score(data_test['labels'], y_pred), 3)}")

    return y_pred, predVI


filetype = 'json'  # 'npz'

def get_data(filename, filetype='npz'):
    """
    :param filename: 
    """
    datapath = os.path.join("example", "data", f"{filename}.{filetype}")
    dataset_path = get_dataset_path(path=datapath)

    match filetype:
        case 'json':
            # Load the JSON file
            with open(dataset_path, "r") as f:
                data = json.load(f)

            X = np.array(data["X"])
            labels = np.array(data["labels"])

        case 'npz':
            # or load the NPZ file
            data = np.load(dataset_path)
            X = data["X"]
            labels = data["labels"]

        case _:
            raise ValueError(f"Unknown dataset type: {filetype}")

    return X, labels


def run(dataset_name="dataset1", filetype="npz"):
    X, labels = get_data(dataset_name, filetype=filetype)

    for tr_size in TR_SIZE:
        logging.info(f"******* training size {tr_size} *******")
        
        classify_data(tr_size, X, labels, random_state=42)

default_input = 'dataset1'
default_filetype = 'npz'

def main():
    parser = argparse.ArgumentParser(description="Run classification on a dataset")
    parser.add_argument("dataset_name", nargs="?", default=default_input,
                        help=f"Name of the dataset folder (default: {default_input})")
    parser.add_argument("--filetype", choices=["npz", "json"], default=default_filetype, help=f"Type of file (default: {default_filetype})")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: Ignored unknown arguments: {unknown}")

    run(dataset_name=args.dataset_name, filetype=args.filetype)

if __name__ == "__main__":
    main()