# Validation segmentation tools

These tools generate geometry-based slab candidates, object predictions and human-reviewed reference masks. The indicator validation uses the supplied masks. The GSA batch has its own inference pipeline.

Run the scripts with the [GSA segmentation environment and weights](../../gsa/README.md).

| Command | Input and output |
|---|---|
| `python slabs.py --image photo.jpg --output slab-results` | Fits floor and ceiling planes separately. Saves support, opening candidates and uncertain regions on the upright image grid. MoGe is the default. Use `--geometry depth-anything` for Depth Anything V2, with `--depth-weights` for a local checkpoint. |
| `python objects.py --images photo.jpg --output object-results` | Uses fixed prompts and image tiles to segment vegetation, light fixtures and signs. Saves class masks, candidate boxes and cropped instance masks. |
| `python signs.py --image photo.jpg --detections object-results --output sign-results` | Classifies sign candidates with Qwen3-VL, then localizes and segments advertising panels. Saves accepted and uncertain masks separately. |
| `python object_references.py --image photo.jpg --annotation photo.json --output references` | Rasterizes LabelMe regions. F denotes advertising, G vegetation and H artificial lighting. An absent label produces an empty reference mask. |
| `python reviewed_walls.py --review review.json --output wall-reference` | Rasterizes reviewed wall polygons within a binary wall domain. The review specifies the source photograph, its upright width and height, and whether the transparent regions are complete. |

The wall review uses paths relative to its JSON file. Polygon labels are `wall_transparent`, `wall_solid`, `wall_uncertain` and `foreground_occlusion`.

```json
{
  "source_photo": "photo.jpg",
  "image_size": [640, 480],
  "wall_domain_mask": "wall_domain.png",
  "transparent_regions_complete": false,
  "polygons": [
    {"label": "wall_transparent", "points": [[40, 70], [170, 70], [170, 220], [40, 220]]}
  ]
}
```

