"""Top-level `vae` package for the dvae distribution.

This file is intentionally minimal to avoid importing heavy optional
dependencies (torch) at package import time. It exposes a stable
`__version__` attribute when available.
"""

try:
    # `setuptools_scm` writes the version to `vae/_version.py` at build time
    from ._version import version as __version__  # type: ignore
except Exception:  # pragma: no cover - best-effort fallback
    __version__ = "0.0.0"

__all__ = ["__version__"]
