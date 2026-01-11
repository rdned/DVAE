# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] – 2026-01-10
### Added
- Initial implementation of the Denoising Variational Auto‑Encoder (DVAE) for binary classification.
- Classifier head built on top of the VAE latent space.
- Modular package structure (`vae.py`, `classifier.py`, `utils/`, `config/`).
- Example datasets (`dataset1`, `dataset2`) in JSON and NPZ formats.
- Example script (`example/tests.py`) for manual validation.
- Reproducible wheel‑based build workflow using a two‑directory setup.
- Pinned dependency versions for deterministic installation.
- Comprehensive README with installation, usage examples, and configuration details.
- Apache‑2.0 LICENSE and NOTICE file for third‑party attribution.
- CONTRIBUTING guidelines for development workflow.

---

## [Unreleased]
### Planned
- Automated test suite.
- Hyperparameter search utilities.
- GPU‑accelerated training examples.
- Additional dataset loaders.
