"""Install the upstream model code and public checkpoints."""
from contextlib import contextmanager
from hashlib import file_digest
import os
from pathlib import Path
import shutil
import tarfile
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
EMSANET = "https://drive.usercontent.google.com/download?id=1LD4_g-jL4KJPRUmCGgXxx2xGQ7TNZ_o2&export=download&confirm=t"
DEPTH = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/5426e4f0f36572d16453bbda7a8389317b1bef99/model.safetensors"
SOURCES = (
    ("TUI-NICR/EMSANet", "9e739fd9abf445349028bb8e812d367362413bec", "emsanet", "emsanet/",
     "1d0c4df677c5777c6244e2620fcdbbb8962879d14d47aa6d1a2f4cae7dc5d236"),
    ("TUI-NICR/nicr-multitask-scene-analysis", "7c214e398c149a1a955f07b8f80317b87c665011",
     "emsanet/lib/nicr-multitask-scene-analysis", "src/",
     "0b428471f3eac1122a23d316cb95ad0bd300500db0ea5ba7aa2fff6826f9ee04"),
    ("TUI-NICR/nicr-scene-analysis-datasets", "7068c9f176d735a676be8e08f362b329885a4751",
     "emsanet/lib/nicr-scene-analysis-datasets", "src/",
     "68b39e62616ab3ecc2c9688e1cef7f94d040ddaba3879326d1d111daa0e0f565"),
    ("EnquanYang2022/PGDENet", "52567dc0921567862caa866a4856b6e008464a80", "pgdenet",
     "toolbox/models/BBSnetmodel/", "463aec714682b788286de3c48c087f8d8b68bc8494f012220dbf17f693f9f39c"),
)


def checksum(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


@contextmanager
def staging(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=directory, delete=False) as stream:
        path = Path(stream.name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def download(url, path, expected):
    request = Request(url, headers={"User-Agent": "GSA-benchmark"})
    with urlopen(request, timeout=120) as response, path.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)
    if checksum(path) != expected:
        raise ValueError(f"Download checksum mismatch for {url}")


def install_source(repo, revision, folder, prefix, digest, archive):
    root = ROOT / "external" / folder
    with ZipFile(archive) as bundle:
        for member in bundle.infolist():
            _, _, relative = member.filename.partition("/")
            if member.is_dir() or not (relative.startswith(prefix) or relative == "LICENSE"):
                continue
            destination = (root / relative).resolve()
            if not destination.is_relative_to(root):
                raise ValueError(f"Invalid archive entry: {relative}")
            data = bundle.read(member)
            if folder == "pgdenet" and relative.endswith("/BBSnet.py"):
                data = data.replace(b"from toolbox.models.BBSnetmodel.ASPP import ASPP", b"")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with staging(destination.parent) as temp:
                temp.write_bytes(data)
                os.replace(temp, destination)
    if folder == "pgdenet":
        for relative in ("toolbox", "toolbox/models"):
            (root / relative / "__init__.py").touch()


def main():
    model = ROOT / "model"
    for source in SOURCES:
        repo, revision, folder, prefix, digest = source
        with staging(model) as archive:
            download(f"https://codeload.github.com/{repo}/zip/{revision}", archive, digest)
            install_source(*source, archive)
    depth = model / "depth" / "model.safetensors"
    expected = "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"
    if not depth.exists() or checksum(depth) != expected:
        with staging(depth.parent) as temp:
            download(DEPTH, temp, expected)
            os.replace(temp, depth)
    emsanet = model / "emsanet.pth"
    expected = "ce38b8514396e2fb44a3c013dab396f1a7ddfb6a4e8ef14def4d9b5768344032"
    if not emsanet.exists() or checksum(emsanet) != expected:
        with staging(model) as archive, staging(model) as temp:
            download(EMSANET, archive, "b1ab4031f7523ac23c83e5b678d0cc56ee56887188a995d8a8049595865c902b")
            with tarfile.open(archive) as bundle:
                with bundle.extractfile("sunrgbd/r34_NBt1D_pre.pth") as src, temp.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
            if checksum(temp) != expected:
                raise ValueError("EMSANet checkpoint checksum mismatch")
            os.replace(temp, emsanet)


if __name__ == "__main__":
    main()
