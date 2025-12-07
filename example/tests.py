import torch
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from vae.vae import VAE
from vae.utils.logger import logger
from vae.config.hyperparameters import N_EPOCHS_EVAL, MINIBATCH_SIZE

import json
import logging
# Set the level for this module
logger.setLevel(logging.INFO)   # or DEBUG, WARNING, ERROR, CRITICAL

# CONFIGURATION
# Check for GPU availability
USE_CUDA = torch.cuda.is_available()
TR_SIZE = [2, 3, 4, 5, 6, 8, 10, 16]


def shuffle_split(*arg, tr_sz=10, random_state=42):
    """
    :param arg: X, labels
    :param tr_sz: training size
    :param random_state:
    :return: dict(encoder=train_2Darray, y=labels),
             dict(encoder=test_2Darray, y=labels)
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

    return dict(encoder=tr_encoder, y=labels[train_idx]), \
           dict(encoder=te_encoder, y=labels[test_idx])


def train(data_train):
    """
    :param data_train: training data
    :return: trained model and optimal threshold for binary classification
    """
    bayesnn = VAE(use_cuda=USE_CUDA, **data_train)
    pred_train = bayesnn.infer_parameters(dt_tr=data_train, num_epochs=N_EPOCHS_EVAL, batch_size=MINIBATCH_SIZE)

    thresholds = np.arange(0.05, 0.95, 0.05)
    accs_tr = np.array([round(accuracy_score(data_train['y'], pred_train >= th), 3) for th in thresholds])
    acc_best = max(accs_tr)
    best_idxs = np.where(accs_tr == acc_best)[0]
    if len(best_idxs) > 1:
        logger.warning(f"Multiple best accuracies: {accs_tr}")
        th_proposed = thresholds[best_idxs].mean()  # chose the average threshold
    else:
        th_proposed = thresholds[best_idxs[0]]

    logger.debug(f"The proposed threshold: {th_proposed}")
    logger.debug(f"The best training accuracy {acc_best} at {best_idxs}, {th_proposed}")

    return bayesnn, th_proposed

def test(bayesnn, threshold, data_test):
    """
    :param bayesnn: trained model
    :param th_proposed: threshold
    :param data_test: test data
    :return: prediction and score in [0,1]
    """
    predVI = bayesnn.calc_predVI(batch_size=MINIBATCH_SIZE, **data_test)
    y_pred = predVI >= threshold

    return y_pred, predVI


def classify(tr_size, X, labels, random_state=42):
    """
    :param tr_size: training size
    :param X: X
    :param labels: labels
    :param random_state:
    :return: prediction and score in [0,1]
    """
    data_train, data_test = shuffle_split(X, labels, tr_sz=tr_size, random_state=random_state)
    bayesnn, th_proposed = train(data_train)
    y_pred, predVI = test(bayesnn, th_proposed, data_test)
    logger.info(f"Test accuracy: {round(accuracy_score(data_test['y'], y_pred), 3)}")

    return y_pred, predVI


# Load the JSON file
#with open("data/dataset.json", "r") as f:
#    data = json.load(f)

#X = np.array(data["X"])
#labels = np.array(data["labels"])

# or load the NPZ file
data = np.load("example/data/dataset.npz")
X = data["X"]
labels = data["labels"]


for tr_size in TR_SIZE:
    logger.info(f"******* training size {tr_size} *******")
    classify(tr_size, X, labels, random_state=42)
