import torch
from torch.utils.data import DataLoader
import scipy.sparse as ss

import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from vae.vae import VAE
from vae.utils.logger import logger
from vae.config.hyperparameters import N_EPOCHS_EVAL, MINIBATCH_SIZE
from vae.utils.utils import get_dataset_path

import json
import logging
import argparse

# Set the level for this module
logger.setLevel(logging.INFO)   # or DEBUG, WARNING, ERROR, CRITICAL

# CONFIGURATION
# Check for GPU availability
USE_CUDA = torch.cuda.is_available()
TR_SIZE = [2, 3, 4, 5, 6, 8, 10, 16]  # training size n


def create_loader(**dta):
    """
        Create a PyTorch DataLoader from dictionary-like input.

        Parameters
        ----------
        dta : dict
            Feature arrays keyed by name. Sparse matrices are converted to dense.
            'labels' treated as integer labels.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader yielding batches as dictionaries {feature_name: tensor}.
        """

    class MyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.features_dict = {
                'encoder': torch.Tensor(
                    dta['encoder'].toarray() if isinstance(dta['encoder'], ss.spmatrix) else dta['encoder']
                ),
                'labels': torch.Tensor(np.array(dta['labels'])).to(torch.int64)}

        def __getitem__(self, index):
            return dict((f_name, f_values[index]) for f_name, f_values in self.features_dict.items())

        def __len__(self):
            return max([len(x) for x in self.features_dict.values()]+[0])

    return DataLoader(MyDataset(), batch_size=MINIBATCH_SIZE, num_workers=0, pin_memory=USE_CUDA, shuffle=False)


def shuffle_split(*arg, tr_sz=10, random_state=42):
    """
    :param arg: X, labels
    :param tr_sz: training size
    :param random_state:
    :return: dict(encoder=train_2Darray, labels=labels),
             dict(encoder=test_2Darray, labels=labels)
    """
    dta = shuffle(*arg, random_state=random_state)
    labels = np.array(dta[-1], dtype=bool)
    train_idx = np.hstack((np.where(labels)[0][:tr_sz], np.where(~labels)[0][:tr_sz]))
    test_idx = np.hstack((np.where(labels)[0][tr_sz:], np.where(~labels)[0][tr_sz:]))

    assert len(dta) == 2, f"Data (X,labels) to split has a strange structure: len(data)={len(dta)}"
    logger.debug("Splitting X and labels.")
    labels = np.squeeze(labels)
    tr_encoder = dta[0][train_idx]
    te_encoder = dta[0][test_idx]

    return dict(encoder=tr_encoder, labels=labels[train_idx]), \
           dict(encoder=te_encoder, labels=labels[test_idx])


def train(train_loader):
    """
    :param train_loader: to load batches of training data
    :return: trained model and prediction scores in [0,1]
    """
    feature_dim = next(iter(train_loader))["encoder"].shape[1]
    bayesnn = VAE(feature_dim)
    score_train = bayesnn.infer_parameters(train_loader, num_epochs=N_EPOCHS_EVAL)

    return bayesnn, score_train


def propose_threshold(score_train, labels_train):
    """
    :param score_train: prediction scores in [0,1]
    :param labels_train: labels of training data
    :return: proposed threshold for binary classification
    """
    thresholds = np.arange(0.05, 0.95, 0.05)
    accs_tr = np.array([round(accuracy_score(labels_train, score_train >= th), 3) for th in thresholds])
    acc_best = max(accs_tr)
    best_idxs = np.where(accs_tr == acc_best)[0]
    if len(best_idxs) > 1:
        logger.warning(f"Multiple best accuracies: {accs_tr}")
        th_proposed = thresholds[best_idxs].mean()  # chose the average threshold
    else:
        th_proposed = thresholds[best_idxs[0]]

    logger.debug(f"The proposed threshold: {th_proposed}")
    logger.debug(f"The best training accuracy {acc_best} at {best_idxs}, {th_proposed}")

    return th_proposed


def test(bayesnn, threshold, test_loader):
    """
    :param bayesnn: trained model
    :param threshold: threshold for binarizing the predictions
    :param test_loader: to load the batches of test data
    :return: predictions and scores in [0,1]
    """

    predVI = bayesnn.calc_predVI(test_loader)
    y_pred = predVI >= threshold

    return y_pred, predVI


def classify_data(tr_size, X, labels, random_state=42):
    """
    :param tr_size: training size n
    :param X: data to classify
    :param labels: labels
    :param random_state:
    :return: predictions and scores in [0,1]
    """
    data_train, data_test = shuffle_split(X, labels, tr_sz=tr_size, random_state=random_state)

    train_loader = create_loader(**data_train)
    bayesnn, pred_train = train(train_loader)
    th_proposed = propose_threshold(pred_train, data_train['labels'])

    test_loader = create_loader(**data_test)
    y_pred, predVI = test(bayesnn, th_proposed, test_loader)
    logger.info(f"Test accuracy: {round(accuracy_score(data_test['labels'], y_pred), 3)}")

    return y_pred, predVI


filetype = 'json'  # 'npz'

def get_data(filename, filetype='npz'):
    """
    :param filename: 
    """
    dataset_path = get_dataset_path(filename=f"{filename}.{filetype}")

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

