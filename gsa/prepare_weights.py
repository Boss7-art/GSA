"""Restore and verify the local model files used by GSA."""
import argparse
from contextlib import closing, contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from model_assets import assets_root


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "weights_manifest.json"
DETECTOR = "detector.safetensors"
BERT = "bert-base-uncased/model.safetensors"
SOURCE_CHECKPOINT = "groundingdino_swint_ogc.pth"


def valid(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]


@contextmanager
def temporary_file(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, prefix=".gsa-", delete=False) as stream:
        path = Path(stream.name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def fetch(entry, destination, source=None):
    if source is not None:
        if not source.is_file():
            raise FileNotFoundError(f"Missing local file: {source}")
        with source.open("rb") as src, destination.open("wb") as out:
            shutil.copyfileobj(src, out, length=1024 * 1024)
    else:
        url = entry["url"]
        if not url.startswith("https://"):
            raise ValueError(f"Expected an HTTPS model URL: {url}")
        request = Request(url, headers={"User-Agent": "GSA-model-fetch/1"})
        with closing(urlopen(request, timeout=120)) as response:
            if not response.geturl().startswith("https://"):
                raise ValueError(f"Non-HTTPS redirect for {url}")
            with destination.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
    if not valid(destination, entry):
        raise ValueError(f"Size or SHA-256 mismatch for {source or entry['url']}")


def restore_detector(manifest, entries, destination, local):
    import torch
    from safetensors.torch import save_file

    print(f"Preparing {SOURCE_CHECKPOINT}", flush=True)
    with temporary_file(destination) as checkpoint:
        local_file = local / SOURCE_CHECKPOINT if local else None
        fetch(manifest["detector_checkpoint"], checkpoint, local_file)
        state = torch.load(checkpoint, weights_only=True, map_location="cpu")["model"]
        for entry in entries:
            relative = entry["path"]
            is_encoder = relative == BERT
            tensors = {}
            for full_key, value in state.items():
                key = full_key.removeprefix("module.")
                if key.startswith("bert.") == is_encoder:
                    name = key.removeprefix("bert.") if is_encoder else key
                    tensors[name] = value.detach().cpu().contiguous().clone()
            target = destination / relative
            with temporary_file(target.parent) as stage:
                save_file(tensors, str(stage), metadata={"format": "pt"})
                if not valid(stage, entry):
                    raise ValueError(f"Converted checkpoint differs from the manifest: {relative}. "
                                     "Use the PyTorch and SafeTensors versions in the installation instructions.")
                os.replace(stage, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Verify all model files without changing or downloading them")
    mode.add_argument("--from-dir", type=Path, metavar="WEIGHTS", help="Use an existing weights directory offline")
    parser.add_argument("--weights-dir", type=Path, default=assets_root(),
                        help="Destination or directory to check (default: GSA_WEIGHTS_DIR or the project's weights directory)")
    args = parser.parse_args()
    destination = args.weights_dir.resolve()
    local = args.from_dir.resolve() if args.from_dir else None
    if local is not None and not local.is_dir():
        parser.error(f"No such weights directory: {local}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    entries = manifest["files"]
    missing = [entry for entry in entries if not valid(destination / entry["path"], entry)]
    if args.check and missing:
        for entry in missing:
            print(f"Missing or incorrect: {entry['path']}", file=sys.stderr)
        return 1
    bundled = {}
    for entry in missing:
        relative = entry["path"]
        if "url" in entry or relative in (DETECTOR, BERT):
            continue
        source = ROOT / "weights" / relative
        if not valid(source, entry):
            if local is None or not valid(local / relative, entry):
                raise ValueError(f"Missing or incorrect GSA model in the repository: weights/{relative}. "
                                 "Restore it from the repository or supply an intact weights directory with --from-dir.")
            source = local / relative
        bundled[relative] = source
    to_convert = []
    for entry in missing:
        relative = entry["path"]
        target = destination / relative
        source = bundled.get(relative, local / relative if local is not None else None)
        if relative in (DETECTOR, BERT) and (source is None or not source.is_file()):
            to_convert.append(entry)
            continue
        print(f"Preparing {relative}", flush=True)
        with temporary_file(target.parent) as stage:
            fetch(entry, stage, source)
            os.replace(stage, target)
    if to_convert:
        restore_detector(manifest, to_convert, destination, local)
    print(f"Verified {len(entries)} model files in {destination}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ImportError, ValueError, HTTPError, URLError) as error:
        sys.exit(f"Weight preparation failed: {error}")
