# DVAE Classifier
[![Python](https://img.shields.io/badge/Python-≤3.12.12-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-<2.0-important)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red)](https://pytorch.org/)
[![Pyro](https://img.shields.io/badge/Pyro-1.9.1-orange)](https://pyro.ai/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

Pyro/PyTorch implementation of the **Denoising Variational Auto‑Encoder (DVAE)** for binary classification in extremely low‑data regimes, based on the paper:

> *Harnessing Variational Auto‑Encoder for Binary Classification in Extremely Low Data Regime*  
> Radim Nedbal and Babak Mahdian (2025), submitted to *Progress in Artificial Intelligence*.

## Overview

DVAE is a neural network designed to classify binary data with very few labeled examples. It combines the power of Variational Auto-Encoders with denoising techniques to achieve robust classification in extremely low-data regimes.

The classifier implementation is centered on the `VAE` class in `dvae/vae.py`.

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

Clone the repository:

```bash
git clone https://github.com/rdned/DVAE.git
cd DVAE
```

Create a virtual environment using `pyenv` and install dependencies:

```bash
pyenv install 3.12.12 --skip-existing
pyenv virtualenv 3.12.12 dvae-3.12.12
pyenv local dvae-3.12.12
```

Update packaging tools:

```bash
pip install --upgrade pip setuptools wheel
```

Install the package (editable mode):

```bash
pip install -e .
```
>This installs the dvae package and all runtime dependencies declared in pyproject.toml.

### Installation from Source

This project uses a two-directory workflow:

- `DVAE/` — the source tree containing the Python package and build configuration
- `dvae_test/` — a clean environment used to install and test the built wheel

#### 1. Build the wheel (inside `DVAE/`)

```bash
cd DVAE
pip install --upgrade build
python -m build
```

This produces:

```
DVAE/dist/dvae-0.1.0-py3-none-any.whl
```

All runtime dependencies (NumPy < 2, PyTorch 2.2.2, Pyro 1.9.1, scikit‑learn 1.7.2) are encoded in the wheel metadata.

#### 2. Create a clean test environment using pyenv (inside `dvae_test/`)

```bash
cd ../dvae_test
pyenv virtualenv 3.12.12 dvae-test
pyenv local dvae-test
```

#### 3. Install the wheel

PyTorch requires NumPy to be present at install time. Install NumPy first, then PyTorch, then the wheel:

```bash
pip install ../DVAE/dist/dvae-0.1.0-py3-none-any.whl
```
Pip will automatically install the correct versions of:
* NumPy (< 2)
* PyTorch 2.2.2 (CPU build from PyPI)
* Pyro 1.9.1
* scikit‑learn 1.7.2

For GPU builds, follow the official PyTorch instructions:
https://pytorch.org/

#### 4. Provide the dataset path explicitly

The dataset is not bundled with the package. You must supply its location explicitly.

Option A — Pass the path directly:

```bash
python - << 'EOF'
from dvae.utils import get_dataset_path
path = get_dataset_path("/absolute/path/to/DVAE/example/data/dataset1.json")
print(path)
EOF
```

Option B — Use an environment variable:

```bash
export DATASET_PATH=/absolute/path/to/DVAE/example/data/dataset1.json
```

Then in Python:

```bash
python - << 'EOF'
from dvae.utils import get_dataset_path
print(get_dataset_path())
EOF
```

## Quick Start

After installing DVAE, you can verify that the package is working by importing it
and checking the version:

```bash
python -c "import dvae; print(dvae.__version__)"
```

>This confirms that the library and its runtime dependencies were installed correctly.
For end‑to‑end usage examples (including dataset loading and model training),
see the Installation from Source section. The example scripts and datasets
are available only in a source checkout and are not included in the wheel.

## Repository Structure

```
.
├── CONTRIBUTING.md                 # Contribution guidelines and dev workflow
├── LICENSE                         # License information
├── README.md                       # Project overview and usage
├── pyproject.toml                  # Project metadata and runtime dependencies
│
├── example/                        # Usage demonstrations and manual validation scripts
│   ├── data/                       # Example datasets
│   │   ├── dataset1.json
│   │   ├── dataset1.npz
│   │   ├── dataset2.json
│   │   └── dataset2.npz
│   └── tests.py                    # Manual example script (not part of automated tests)
│
└── src/                            # Source layout root (contains only packages)
    └── dvae/                       # Main DVAE package (imported as `import dvae`)
        ├── __init__.py             # Package entry point
        ├── _version.py             # Version management
        ├── classifier.py           # Classifier built on top of the VAE
        ├── vae.py                  # Core VAE implementation
        │
        ├── config/                 # Configuration modules (internal)
        │   ├── __init__.py
        │   └── hyperparameters.py  # Default hyperparameter definitions
        │
        └── utils/                  # Utility functions (internal)
            ├── __init__.py
            ├── custom_mlp.py       # Custom MLP architecture used by the VAE
            ├── logger.py           # Lightweight logging utilities
            └── utils.py            # Miscellaneous helpers
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

Training and model hyperparameters are controlled in `dvae/config/hyperparameters.py`.

### Key Configuration Sections

* Training: Learning rate, decay rates, batch size, number of epochs
* DVAE:
  * Objective function parameters (KL divergence weight, reconstruction loss weight, classification weight)
  * Bernoulli corruption rate (noise added during training)
  * Architectural parameters (hidden layer sizes, latent dimension)

For detailed configuration options, see `dvae/config/hyperparameters.py`.


## Dependencies

This project pins exact versions for reproducibility. Key dependencies include:

* numpy 1.26.4 – Numerical computing (**required: PyTorch 2.2.2 is not compatible with NumPy ≥ 2.0**)  
* torch 2.2.2 – Deep learning framework  
* pyro-ppl 1.9.1 – Probabilistic programming library  
* scikit-learn 1.7.2 – Machine learning utilities

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
