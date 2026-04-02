# Contributing

Thanks for your interest in contributing to DVAE.

## Development setup

Create a local virtual environment and install the package for development.

* Using `pyenv virtualenv` is described in Subsect. Setup in Sect. Installation in README.md.
* Alternatively, using `venv`:

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

To build a wheel and source distribution locally refer to README.md Sect. Installation from Source. In brief:

```bash
pip install --upgrade build
python -m build
# install the built wheel
pip install dist/*.whl
```

## Release & tagging workflow (setuptools_scm)

>DVAE uses setuptools_scm, which derives the version entirely from Git tags.
>No version numbers are stored in the source tree.
>Every release must follow this deterministic workflow.

### 1. Ensure a cleen working tree

```bash
git status -s
```

### 2. Choose the next version

```vMAJOR.MINOR.PATCH```, e.g., ```v0.2.1```

### 3. Commit all release‑relevant changes

```bash
git add -A
git commit -m "Release v0.1.2"
```

### 4. Tag the release commit

* Tags must point to HEAD, not a previous commit.

```bash
git tag v0.1.2
```

* Verify the tag:

```bash
git show v0.1.2
git rev-parse HEAD
```

>The commit hashes must match.


### 5. Build the release artifacts

```bash
rm -rf dist/
python -m build
```

* Expected output in `dist/`:

```
dvae-0.1.2-py3-none-any.whl
dvae-0.1.2.tar.gz
```

>If you see a version like 0.1.3.dev0+gHASH, the working tree is dirty or the tag is not on HEAD.

### 6  (Optional) Upload to TestPyPI

```bash
pip install --upgrade twine
twine upload --repository testpypi dist/*
```

* Test installation in a fresh virtual environment:

```bash
cd new_test_dir
python -m venv .venv-test
source .venv-test/bin/activate
pip install \
  --index-url https://test.pypi.org/simple \
  --extra-index-url https://pypi.org/simple \
  dvae==0.1.2
```

### 7. Upload to PyPI

```bash
twine upload dist/*
```

### 8. Push the release commit and tag

```bash
git push
git push --tags
```
