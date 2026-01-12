# Contributing

Thanks for your interest in contributing to DVAE.

## Development setup

Create a local virtual environment and install the package for development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install test tools and common dev utilities (if not already available):

```bash
pip install pytest
# optional: black, flake8, isort
pip install black flake8 isort
```

## Tests

Run the unit tests with `pytest`:

```bash
pytest -q
```

To run only the example tests (these require example dataset files):

```bash
# Run the example test runner against dataset1 (NPZ files):
python -m example.tests dataset1 --filetype npz
```

Note: Example tests expect the example dataset files to be present under `example/data/` (or set the `DATASET_PATH` env var to the dataset root). In CI we skip these tests because datasets are external; to include them provide a stable test dataset or mock the dataset path in the workflow.

## Saving & persistence

A recommended, robust way to persist trained models is via the classifier helpers:

```python
# save state-dict + metadata (CPU-safe)
clf.save("clf.pth")

# load on CPU and optionally move to GPU
clf2 = VAEClassifier.load("clf.pth", device="cpu")
```

Avoid pickling the whole model object with `pickle.dump(model, f)` as it is fragile and often fails due to non-pickleable internals (PyTorch/Pyro/logging objects). If needed, `dill` can sometimes succeed but it is less portable.

## Contribution workflow

1. Create a feature branch: `git checkout -b feature/short-description`.
2. Add or update tests for your change.
3. Run the test suite locally and fix any issues.
4. Update `CHANGELOG.md` or add a short note to `docs/` describing user-visible changes.
5. Commit with a clear message (e.g., `feat: add save/load helpers`) and push the branch.
6. Open a Pull Request against `main` and request at least one reviewer.

## Formatting & checks

- We recommend running `black` and `isort` to format changes before committing.
- If you add static checks (mypy/flake8), include instructions here and add them to CI where appropriate.

## Building locally

To build a wheel and source distribution locally:

```bash
pip install --upgrade build
python -m build
# install the built wheel
pip install dist/*.whl
```

## Publishing

We do not publish automatically. Use `twine` to upload to TestPyPI or PyPI after creating an account and generating an API token.

```bash
pip install --upgrade twine
twine upload --repository testpypi dist/*
# test installation
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple dvae
```
