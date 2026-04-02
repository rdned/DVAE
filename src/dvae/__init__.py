"""Top-level `dvae` package for the dvae distribution.

This file is intentionally minimal to avoid importing heavy optional
dependencies (torch) at package import time. It exposes a stable
`__version__` attribute when available.
"""

import sys

if sys.version_info >= (3, 12):
    raise RuntimeError(
        "DVAE requires Python < 3.12 due to PyTorch 2.2.2 incompatibility "
        "(use 3.10 or 3.11)."
    )

try:
    # `setuptools_scm` writes the version to `vae/_version.py` at build time
    from ._version import version as __version__  # type: ignore
except Exception:  # pragma: no cover - best-effort fallback
    __version__ = "0.0.0"


from .classifier import VAEClassifier

__all__ = ["VAEClassifier", "__version__"]
