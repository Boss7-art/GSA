import argparse
import json
import math
import statistics
from pathlib import Path

from PIL import Image


SLAB_MASKS = ("floor_opening", "ceiling_opening", "floor_solid", "ceiling_solid")
WALL_MASKS = ("wall_transparent", "wall_solid")
INDICATORS = (
    ("slab_void_to_solid", "Slab void-to-solid ratio", ("A", "B"), SLAB_MASKS),
    ("wall_void_to_solid", "Wall void-to-solid ratio", ("C", "D"), WALL_MASKS),
    ("enclosure_degree", "Enclosure degree", ("A", "B", "C", "D"), SLAB_MASKS + WALL_MASKS),
    ("signage_visual_prominence", "Signage visual prominence", ("F",), ("F",)),
    ("natural_elements", "Natural elements", ("G",), ("G",)),
    ("artificial_lighting", "Artificial lighting", ("H",), ("H",)),
)


def count_pixels(directory, labels, size):
    counts = {}
    for label in labels:
        path = directory / f"{label}_mask.png"
        with Image.open(path) as mask:
            if mask.mode not in {"1", "L"} or mask.size != size:
                raise ValueError(f"Invalid mask dimensions or mode: {path}")
            pixels = mask.convert("L").histogram()
        if pixels[0] + pixels[255] != math.prod(size):
            raise ValueError(f"Expected a binary mask: {path}")
        counts[label] = pixels[255]
    return counts


def photo_values(kind, manual, machine, size):
    if kind == "slab_void_to_solid":
        return (manual["B"] / manual["A"],
                (machine["floor_opening"] + machine["ceiling_opening"]) /
                (machine["floor_solid"] + machine["ceiling_solid"]))
    if kind == "wall_void_to_solid":
        return manual["D"] / (manual["C"] - manual["D"]), machine["wall_transparent"] / machine["wall_solid"]
    if kind == "enclosure_degree":
        return ((manual["B"] + manual["D"]) / (manual["A"] + manual["C"] - manual["D"]),
                (machine["floor_opening"] + machine["ceiling_opening"] + machine["wall_transparent"]) /
                (machine["floor_solid"] + machine["ceiling_solid"] + machine["wall_solid"]))
    label = next(iter(manual))
    return manual[label] / math.prod(size), machine[label] / math.prod(size)


def evaluate(root, kind, name, manual_labels, machine_labels):
    rows = []
    folder = root / "indicators" / kind
    for photo in sorted(folder.iterdir()):
        if not photo.is_dir():
            continue
        annotation = json.loads((photo / "manual" / "annotation.json").read_text(encoding="utf-8"))
        size = annotation["imageWidth"], annotation["imageHeight"]
        manual = count_pixels(photo / "manual", manual_labels, size)
        machine = count_pixels(photo / "machine", machine_labels, size)
        x, y = photo_values(kind, manual, machine, size)
        rows.append({"photo_id": photo.name, "manual_value": x, "machine_value": y,
                     "manual_mask_pixels": manual, "machine_mask_pixels": machine})

    manual_values = [row["manual_value"] for row in rows]
    machine_values = [row["machine_value"] for row in rows]
    nmae = 100 * math.fsum(abs(x - y) for x, y in zip(manual_values, machine_values)) / math.fsum(manual_values)
    pearson = statistics.correlation(manual_values, machine_values)
    return {"indicator": name, "n": len(rows), "NMAE_percent": nmae, "Pearson_r": pearson, "photo_values": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [evaluate(args.archive, *metric) for metric in INDICATORS]
    for result in results:
        print(f"{result['indicator']}: n={result['n']}, NMAE={result['NMAE_percent']:.2f}%, Pearson r={result['Pearson_r']:.4f}")
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump({"metrics": results}, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
