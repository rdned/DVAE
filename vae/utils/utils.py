import math
from pathlib import Path
import os


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)



def get_dataset_path(filename: str = "dataset.npz") -> Path:
    """
    Resolve the dataset path relative to the repo root and provide diagnostics
    if the file is missing.
    """
    # climb up from vae/utils/utils.py to repo root
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "example" / "data"
    dataset_path = data_dir / filename

    if not dataset_path.exists():
        cwd = Path.cwd()
        raise FileNotFoundError(
            f"\n[Dataset diagnostic]\n"
            f"  Expected path : {dataset_path}\n"
            f"  Current CWD   : {cwd}\n"
            f"  sys.path[0]   : {os.sys.path[0]}\n"
            f"Ensure you run from project root ({project_root}) "
            f"or check that {filename} exists in example/data."
        )
    return dataset_path


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