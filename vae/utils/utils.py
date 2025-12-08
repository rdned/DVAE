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


