import os
from pathlib import Path


def assets_root():
    configured = os.environ.get("GSA_WEIGHTS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent / "weights"
