# DVAE Usage Guide

This document provides a clear, minimal, and reproducible introduction to using the DVAE package. Every example is self‑contained and can be copied directly into a Python session.

---

## 1. Installation

Install DVAE:

```bash
pip install dvae
```

Verify installation:

```python
import dvae
print("DVAE version:", dvae.__version__)
```

---

## 2. Core Concepts

DVAE provides:

- **VAEClassifier** — a variational autoencoder with a classifier head  
- **Configuration system** — structured hyperparameters with explicit provenance  
- **Utilities** — helpers for training, evaluation, and reproducibility  

The public API is intentionally small and explicit.

---

## 3. Quickstart Example

```python
from dvae import VAEClassifier

# Preferred: explicitly pass VAE args expected by code
model = VAEClassifier(feature_dim=10, z_dim=2)

# Or use hyperparameters:
from dvae.config import hyperparameters
h = hyperparameters.default_hparams()
h.input_dim = 10
h.z_dim = 2
model = VAEClassifier.from_hparams(h)
```

---

## 4. Using Hyperparameters

```python
from dvae.config import hyperparameters

hparams = hyperparameters.default_hparams()
print(hparams)

# Note: prefer canonical names such as `z_dim` and `input_dim`.
# Aliases (e.g., `latent_dim`) are available for convenience.
```

Override values:

```python
# Use canonical names where possible; aliases also work.
hparams.z_dim = 4  # canonical name (alias: `latent_dim`)
hparams.learning_rate = 1e-3
```

---

## 5. Creating a Model from Hyperparameters

```python
from dvae import VAEClassifier
from dvae.config import hyperparameters

hparams = hyperparameters.default_hparams()
hparams.input_dim = 20
hparams.z_dim = 3  # prefer canonical `z_dim` (alias: `latent_dim`)

model = VAEClassifier.from_hparams(hparams)
```

---

## 6.Fitting the Classifier

```python
import numpy as np

X = np.random.randn(200, 20)
y = np.random.randint(0, 2, size=200)

model.fit(X, y, num_epochs=200)

# Note: `VAEClassifier` currently supports only binary classification (two classes).
# `predict()` will raise an error otherwise.
```

---

## 7. Predicting Probabilities

```python
probs = model.predict_proba(X)
print(probs[:5])
```

---

## 8. Predicting Labels

```python
labels = model.predict(X)
print(labels[:5])
```

---

## 9. Transforming Data

* `transform` returns the same output as `predict_proba` (probabilities, not embeddings):

```python
probs = model.transform(X)
print(probs[:5])

# To access embeddings and reconstructed inputs, call `predict_proba`
# then read `model.vae.z_loc_embedding` and `model.vae.x_reconst`:
probs = model.predict_proba(X)
print(model.vae.z_loc_embedding[:5])  # embeddings
print(model.vae.x_reconst[:5])        # reconstructions
```

---

## 10. Accessing the Underlying VAE (forward pass)

```python
vae = model.vae
```

* Forward pass through the VAE:

```python
import torch

x = torch.randn(8, model.feature_dim)
out = vae(x)

print(out.keys())
# dict_keys(['z_loc', 'z_scale', 'y_loc', 'x_recon'])
```

---

## 11. Saving and Loading

* Save the classifier (recommended): prefer the `save` helper which writes a CPU‑safe state dict and metadata. Avoid pickling the entire classifier object with `pickle.dump` (fragile and often fails due to non‑pickleable internals).

```python
# Preferred: saves state_dict + metadata in a portable way
clf.save("clf.pth")
```

* Save the underlying VAE weights (recommended):

```python
import torch
torch.save(model.vae.state_dict(), "vae.pt")
```

* Load (map to CPU or chosen device):

```python
state = torch.load("vae.pt", map_location="cpu")
model.vae.load_state_dict(state)
```

* Preferred: use the classifier helpers to save and load the full classifier state:

```python
# Save full classifier (state dict + metadata)
clf.save("clf.pth")

# Load (placed on CPU)
clf2 = VAEClassifier.load("clf.pth", device="cpu")
```

---

## 12. Reproducibility

```python
from dvae.utils import set_seed
set_seed(42)
```

---

## 13. Full Workflow Example

```python
from dvae import VAEClassifier
from dvae.config import hyperparameters
from dvae.utils import set_seed
import numpy as np
import torch

set_seed(123)

# Hyperparameters
h = hyperparameters.default_hparams()
h.input_dim = 10
h.z_dim = 4  # prefer canonical `z_dim` (alias: `latent_dim`) 

# Classifier
clf = VAEClassifier.from_hparams(h)

# Data
X = np.random.randn(100, 10)
y = np.random.randint(0, 2, size=100)

# Train
clf.fit(X, y, num_epochs=100)

# Predict
probs = clf.predict_proba(X)
labels = clf.predict(X)

# VAE forward pass
vae = clf.vae
x = torch.randn(5, 10)
out = vae(x)
print(out.keys())
```

---

## 14. Troubleshooting

**ImportError: cannot import name 'VAEClassifier'**  
Reinstall DVAE:

```bash
pip install --upgrade dvae
```

**Shape mismatch errors**  
Ensure `input_dim` matches your data.

**CUDA not used**  
`VAE` moves itself to CUDA during construction when available. To manually move, do:

```python
model.vae.cuda()
x = x.cuda()
```

**Binary classification only**  
`VAEClassifier` currently supports only binary classification (two classes). `predict()` will raise an error otherwise.

---

## 15. License

DVAE is released under the Apache License 2.0.  
See the `LICENSE` file for the full text.