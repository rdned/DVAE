# Contributing

Thanks for your interest in contributing to DVAE.

## Tests
Run the example tests in `example/` for quick checks:

```bash
python -m example.test_threshold
python -m example.test_fit_predict
python -m example.test_fit_transform
python -m example.test_fit_predict_on
python -m example.test_fit_predict_proba_on
```

## Building locally
To build a wheel and source distribution locally:

```bash
pip install --upgrade build
python -m build
# install the built wheel
pip install dist/dvae-0.1.0-py3-none-any.whl
```

## Publishing
We do not publish automatically. Use `twine` to upload to TestPyPI or PyPI after creating an account and generating an API token.

```bash
pip install --upgrade twine
twine upload --repository testpypi dist/*
# test installation
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple dvae
```
