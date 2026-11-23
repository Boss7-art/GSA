import hashlib
import os
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def weights():
    default = Path(__file__).resolve().parents[2] / "gsa" / "weights"
    return Path(os.environ.get("GSA_WEIGHTS_DIR", default)).expanduser()
