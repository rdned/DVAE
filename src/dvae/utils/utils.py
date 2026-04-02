import math
from pathlib import Path
import os


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def get_dataset_path(path: str | Path | None = None) -> Path:
    """
    Resolve dataset path from explicit argument or DATASET_PATH env var.
    """
    if path is None:
        env = os.getenv("DATASET_PATH")
        if env is None:
            raise FileNotFoundError(
                "Dataset path not provided.\n"
                "Pass a path explicitly or set DATASET_PATH."
            )
        path = env

    dataset_path = Path(path).expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"\n[Dataset diagnostic]\n"
            f"  Provided path : {dataset_path}\n"
            f"  Current CWD   : {Path.cwd()}\n"
            f"Dataset must be supplied explicitly because it is not packaged.\n"
            f"Check the path or pass a correct one."
        )

    return dataset_path


import numpy as np
import torch
import pyro


def set_seed(seed: int):
    """
    Set RNG seeds for numpy, torch, and pyro.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        pyro.set_rng_seed(seed)
    except Exception:
        pass
