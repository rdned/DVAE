"""
Hyperparameters for DVAE.

This module provides:
- semantic grouping of hyperparameters
- a single dataclass `Hyperparameters` for explicit provenance
- default values matching the original constants
"""

from dataclasses import dataclass

# -----------------------------
# Original constants (preserved)
# -----------------------------

ROUNDING_DIGITS = 4

# Training
LR_ADAM = 5e-4
BETAS = (0.97, 0.999)
N_EPOCHS_EVAL = 1000
N_EPOCHS_TRAIN = 100
MINIBATCH_SIZE = 32
SUBSAMPLE_RATIO = 2

# DVAE objective
ALPHA1 = 1e-5
ALPHA2 = 1
BETA = 1e-2

# Corruption
BCP = 0.7

# Architecture
HIDDEN_DIM = 80
Z_DIM = 80


# -----------------------------
# Structured config
# -----------------------------

from dataclasses import dataclass

@dataclass
class Hyperparameters:
    # -----------------------------
    # Training
    # -----------------------------
    lr_adam: float = LR_ADAM
    learning_rate: float = LR_ADAM  # alias

    betas: tuple = BETAS
    adam_betas: tuple = BETAS  # alias

    n_epochs_eval: int = N_EPOCHS_EVAL
    eval_epochs: int = N_EPOCHS_EVAL  # alias

    n_epochs_train: int = N_EPOCHS_TRAIN
    train_epochs: int = N_EPOCHS_TRAIN  # alias

    minibatch_size: int = MINIBATCH_SIZE
    batch_size: int = MINIBATCH_SIZE  # alias

    subsample_ratio: int = SUBSAMPLE_RATIO
    subsample: int = SUBSAMPLE_RATIO  # alias

    # -----------------------------
    # DVAE objective
    # -----------------------------
    alpha1: float = ALPHA1
    alpha_kl: float = ALPHA1  # alias

    alpha2: float = ALPHA2
    alpha_recon: float = ALPHA2  # alias

    beta: float = BETA
    beta_weight: float = BETA  # alias

    # -----------------------------
    # Corruption
    # -----------------------------
    bcp: float = BCP
    corruption_prob: float = BCP  # alias

    # -----------------------------
    # Architecture
    # -----------------------------
    hidden_dim: int = HIDDEN_DIM
    input_dim: int = HIDDEN_DIM  # alias

    z_dim: int = Z_DIM
    latent_dim: int = Z_DIM  # alias

    seed: int = None

def default_hparams() -> Hyperparameters:
    """Return a fresh Hyperparameters object with default values."""
    return Hyperparameters()
