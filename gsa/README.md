# GSA indoor spatial measurement

GSA segments solids and openings in the floor, ceiling and walls visible from a single storey and corrects their areas for perspective. The full batch also measures seating, vegetation, signage, artificial lighting, enclosure and colour.

## Installation

From this directory, use Python 3.12 and PowerShell to create separate environments for segmentation and scene indicators.

```powershell
py -3.12 -m venv .venv/segmentation
py -3.12 -m venv .venv/indicators
$seg = (Resolve-Path .venv/segmentation/Scripts/python.exe).Path
$ind = (Resolve-Path .venv/indicators/Scripts/python.exe).Path
$torchIndex = 'https://download.pytorch.org/whl/cu128'
& $seg -m pip install torch==2.11.0 torchvision==0.26.0 --index-url $torchIndex
& $ind -m pip install torch==2.11.0 torchvision==0.26.0 --index-url $torchIndex
& $seg -m pip install -r requirements/segmentation.txt
& $seg -m pip install --no-deps "https://github.com/Microsoft/MoGe/archive/74fbce054ebed49800de42d0ad0e83495065719a.zip" "https://github.com/EasternJournalist/utils3d-moge/archive/62f09d58509485564e24d5d9f6aac9ee9ebc0c37.zip"
& $ind -m pip install -r requirements/indicators.txt
& $ind -m pip install --no-deps groundingdino-py==0.4.0 segment-anything==1.0
& $seg prepare_weights.py
```

For a CPU installation, set `$torchIndex` to `https://download.pytorch.org/whl/cpu`. Weight preparation downloads the pinned dependencies and verifies every file against its SHA-256 digest. The complete weights occupy 8.56 GB, with extra space needed during conversion. To reuse a complete weights directory, set `GSA_WEIGHTS_DIR` to it and run `prepare_weights.py --check`. [Weight manifest](weights_manifest.json) / [Model licences](THIRD_PARTY.md)

## Run

```powershell
& $ind cal_ceil.py --input_image ../photos --output_dir results --slab-python $seg
```

Input may be one image or a directory. GSA follows the EXIF orientation and preserves subdirectory names. Use `--slab-only` or `--wall-only` with the segmentation interpreter to run a single model. GPU segmentation is automatic when available. Use `--slab-device cpu` to select the CPU.

## Outputs

| Output | Contents |
|---|---|
| `<image>.txt` | Image-plane indicators, rectified ratios and source pixel coverage. Colour fractions are ordered warm, cool, neutral. |
| `slab_transparency/<image>/classes.png` | 0 unassigned, 1 floor solid, 2 floor opening, 3 ceiling solid, 4 ceiling opening. |
| `wall_transparency/<image>/classes.png` | 0 unassigned, 1 wall solid, 2 glazing or open frontage. |
| `perspective/<image>/result.json` | Plane transformations, reference domains, areas, units and rectified mask paths. |

[The scene indicator module](indicators.py) uses Grounding DINO to detect objects and Segment Anything (SAM) to segment each detection. The full batch sums the mask pixel counts per category, divides by the full image pixel count and writes the fraction to `<image>.txt`. These object masks are used in memory.

| Category | Detector term | Report field |
|---|---|---|
| Seating | `chair` | `seating` |
| Vegetation | `plant` | `green_view_index` |
| Signage | `billboard` | `billboards` |
| Artificial lighting | `light` | `artificial_lighting` |

`slab_transparency` combines the image-plane floor and ceiling opening fractions by pixel area. For each physical plane, `explicit_*` ratios use the same transformation for the segmented openings and solids. With rectified opening area `V` and solid area `S`, `void_over_total` is `V/(V+S)` and `void_over_solid` is `V/S`. When coverage is partial, `explicit_assigned_*` reports the ratio for the measured portion and `*_source_pixel_coverage` gives its coverage for each class. The `rectified_*` opening ratios use the fitted reference area minus its solid area. Missing values appear as `nan` in text and `null` in JSON.

## Scale and plane controls

The default camera height is 1.5 m above the floor. Use `--camera-height-m` to supply a measured height or `--no-camera-scale` to retain relative units. An upward view needs an additional scale reference if the floor cannot be identified. To supply measured rectangular controls, pass a JSON file to `--plane-config`.

```json
{
  "images": {
    "photo.jpg": {
      "mode": "manual",
      "planes": [{
        "id": "wall_1",
        "role": "wall",
        "quad_tl_tr_br_bl": [[120, 100], [500, 120], [480, 430], [130, 400]],
        "width_height_metres": [4, 3]
      }]
    }
  }
}
```

Image keys are relative to the input directory. List the four corners clockwise from the top left in EXIF-upright pixels. A plane may also specify a polygon (`footprint`) or binary mask (`reference_mask`). To remeasure existing segmentations, run the segmentation interpreter with `--perspective-only`, using the same input and output directories.
