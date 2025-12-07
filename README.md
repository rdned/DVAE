# DVAE Classifier
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red)](https://pytorch.org/)
[![Pyro](https://img.shields.io/badge/Pyro-%E2%89%A5%201.7.0-orange)](https://pyro.ai/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

Pyro/PyTorch implementation of the **Denoising Variational Auto‑Encoder (DVAE)** for binary classification in extremely low‑data regimes, based on the paper:

> *Harnessing Variational Auto‑Encoder for Binary Classification in Extremely Low Data Regime*  
> Radim Nedbal and Babak Mahdian (2025), submitted to *Progress in Artificial Intelligence*.

The classifier implementation is centered on the VAE class in `vae/vae.py`.

## Table of Contents
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Examples](#examples)
- [Configuration](#configuration)
- [Citation](#citation)
- [License](#license)


## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/rdned/DVAE.git
cd DVAE
pip install -r requirements.txt
```

## Repository structure

- `vae/vae.py` – main DVAE classifier implementation (class `VAE`)
- `vae/config/hyperparameters.py` – configuration and hyperparameters
- `vae/utils/` – utility modules
  - `__init__.py`
  - `custom_mlp.py`
  - `logger.py`
  - `utils.py`

## Examples

The repository includes example/data with a sample dataset in JSON and NPZ formats:
- `example/data` – data folder, which contains example datafile in json and npz format
- `example/test.py` - script for running the example

Expected conventions:

NPZ format:
* A .npz archive containing arrays:
  * X : array-like, shape (N, ...) input features (e.g., flattened vectors or images)
  * labels : binary labels, shape (N,)
    
JSON format:
```bash
{
  "X": list of lists (n-dimensional feature vectors),
  "labels": list of binary labels (0/1).
}
```

* example/test.py:
   * Loads the the example file, instantiates the model (with config), and runs a short train and eval pass for a list of training sizes.

If you adapt your own dataset:
   * Ensure X and y arrays are aligned and that y contains binary labels (0/1).

## Configuration

Configuration file:

`vae/config/hyperparameters.py` — controls training hyperparameters and DVAE hyperparameters.

Config sections
  * Training: learning rate, decay rates
  * DVAE: objective function parameters, Bernoulli corruption, architectural parameters


## Citation

If you use this code, please cite:

Radim Nedbal and Babak Mahdian (2025).  
Harnessing Variational Auto-Encoder for Binary Classification in Extremely Low Data Regime.  
Submitted to Progress in Artificial Intelligence.  
Code available at: https://github.com/rdned/DVAE.git

BibTeX:

```bibtex
@misc{nedbal2025dvae,
  author       = {Radim Nedbal and Babak Mahdian},
  title        = {Harnessing Variational Auto-Encoder for Binary Classification in Extremely Low Data Regime},
  year         = {2025},
  howpublished = {\url{https://github.com/rdned/DVAE.git}},
  note         = {Submitted to Progress in Artificial Intelligence}
}
```

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
