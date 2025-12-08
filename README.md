# DVAE Classifier
[![Python](https://img.shields.io/badge/Python-≤3.12.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red)](https://pytorch.org/)
[![Pyro](https://img.shields.io/badge/Pyro-1.9.1-orange)](https://pyro.ai/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

Pyro/PyTorch implementation of the **Denoising Variational Auto‑Encoder (DVAE)** for binary classification in extremely low‑data regimes, based on the paper:

> *Harnessing Variational Auto‑Encoder for Binary Classification in Extremely Low Data Regime*  
> Radim Nedbal and Babak Mahdian (2025), submitted to *Progress in Artificial Intelligence*.

## Overview

DVAE is a neural network designed to classify binary data with very few labeled examples. It combines the power of Variational Auto-Encoders with denoising techniques to achieve robust classification in extremely low-data regimes.

The classifier implementation is centered on the `VAE` class in `vae/vae.py`.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Citation](#citation)
- [License](#license)

## Installation

### Prerequisites

- Python 3.12.x (PyTorch does not yet provide wheels for Python 3.13+)
- `pyenv` (recommended for Python version management)

### Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/rdned/DVAE.git
cd DVAE
pyenv install 3.12.12
pyenv virtualenv 3.12.12 dvae-3.12.12
pyenv local dvae-3.12.12
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Quick Start

To verify your installation, run the included example:

```bash
python -m example.tests
```

## Repository Structure

```
├── LICENSE
├── README.md
├── requirements.txt
├── example/
│   ├── data/                    # Example datasets (JSON and NPZ formats)
│   │   ├── dataset1.json
│   │   ├── dataset1.npz
│   │   ├── dataset2.json
│   │   └── dataset2.npz
│   └── tests.py                 # Example training and evaluation script
└── vae/
    ├── config/
    │   └── hyperparameters.py   # Configuration and hyperparameters
    ├── utils/
    │   ├── __init__.py
    │   ├── custom_mlp.py
    │   ├── logger.py
    │   └── utils.py
    └── vae.py                    # Main DVAE classifier implementation
```

## Usage Examples

### Data Formats

#### NPZ Format

* A .npz archive containing:
  * X: array-like, shape (N, D) - input features (e.g., flattened vectors or images)
  * labels: array-like, shape (N,) – binary labels (0 or 1)

#### JSON Format

```json
{
  "X": [[feature_vector_1], [feature_vector_2], ...],
  "labels": [0, 1, 0, ...]
}
```

### Running the Example

To train and test the DVAE classifier with a sample dataset:

```bash
python -m example.tests <dataset_name> --filetype <json|npz>
```

Example:

```bash
python -m example.tests dataset1 --filetype npz
```

This will:

1. Load the example dataset
2. Instantiate the DVAE model with default configuration
3. Run training and evaluation across multiple training set sizes
4. Print corresponding accuracies

### Using Your Own Dataset

1. Prepare your data in JSON or NPZ format (see Data Format section above)
2. Ensure X and labels are aligned and labels are binary (0/1)
3. Run:

```bash
python -m example.tests your_dataset --filetype npz
```

## Configuration

Training and model hyperparameters are controlled in `vae/config/hyperparameters.py`.

### Key Configuration Sections

* Training: Learning rate, decay rates, batch size, number of epochs
* DVAE:
  * Objective function parameters (KL divergence weight, reconstruction loss weight, classification weight)
  * Bernoulli corruption rate (noise added during training)
  * Architectural parameters (hidden layer sizes, latent dimension)

For detailed configuration options, see `vae/config/hyperparameters.py`.


## Dependencies

This project pins exact versions for reproducibility. Key dependencies include:

* torch 2.2.2 – Deep learning framework
* pyro-ppl 1.9.1 – Probabilistic programming library
* scikit-learn 1.7.2 – Machine learning utilities
* numpy 1.26.4 – Numerical computing

## Citation

If you use this code, please cite:

### APA Format:

Radim Nedbal and Babak Mahdian (2025).  
Harnessing Variational Auto-Encoder for Binary Classification in Extremely Low Data Regime.  
Progress in Artificial Intelligence (submitted).  

### BibTeX:

```bibtex
@article{nedbal2025dvae,
  author       = {Nedbal, Radim and Mahdian, Babak},
  title        = {Harnessing Variational Auto-Encoder for Binary Classification in Extremely Low Data Regime},
  journal      = {Progress in Artificial Intelligence},
  year         = {2025},
  note         = {Submitted},
  url          = {https://github.com/rdned/DVAE}
}
```

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
