"""Evaluate EMSANet and PGDENet on the annotated photographs."""
import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run(script, *args):
    subprocess.run([sys.executable, "-B", str(ROOT / "tools" / script), *map(str, args)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("both", "emsanet", "pgdenet"), default="both")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    output = args.output.resolve()
    split = ROOT / "data" / "splits"
    run("prepare_photo_masks_dataset.py", "--input-root", "data/annotations", "--output-root", output,
        "--create-dummy-depth", "--overwrite")
    if args.model in ("both", "emsanet"):
        run("generate_depthanything_depths.py", "--prepared-root", output,
            "--model-id", "model/depth", "--device", "cpu", "--overwrite")
        for variant in ("zero", "direct", "inverted"):
            name = f"emsanet_{variant}"
            depth = [] if variant == "zero" else [
                "--depth-dir", output / "emsanet_samples" / f"depth_anything_v2_small_sunrgbd_stats_{variant}"
            ]
            raw = output / "predictions" / f"{name}_raw"
            run("run_emsanet_on_prepared.py", "--prepared-root", output, "--device", "cpu",
                "--output-subdir", raw.name, *depth, "--overwrite")
            run("package_prediction_masks.py", "--prepared-root", output, "--semantic-dir", raw / "semantic_id",
                "--instance-dir", raw / "instance_id", "--output-subdir", name, "--overwrite")
            run("compute_hungarian_miou.py", "--prepared-root", output, "--prediction-subdir", name)
    if args.model in ("both", "pgdenet"):
        run("finetune_pgdenet_ah_semantic.py", "--prepared-root", output, "--depth-dir", "data/validation_depth",
            "--train-list", split / "train.txt", "--eval-list", split / "validation.txt",
            "--resume", "model/pgdenet.safetensors", "--output-subdir", "pgdenet", "--epochs", "0",
            "--device", "cpu", "--overwrite")
        run("compute_hungarian_miou.py", "--prepared-root", output, "--prediction-subdir", "pgdenet",
            "--sample-list", split / "validation.txt")


if __name__ == "__main__":
    main()
