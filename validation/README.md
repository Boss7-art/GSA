# Indicator validation

From this directory, run:

```sh
python -m pip install -r requirements.txt
python recalculate_table8.py
```

Table 8 uses the supplied image-plane masks and indicator definitions. Its enclosure comparison divides the combined area labelled as openings by the combined solid area. The script reports normalized mean absolute error and Pearson correlation for six indicators across 35 indicator–photograph pairs from 24 photographs.

The calculation works from the masks alone. To display the polygons over their photographs, extract the [segmentation benchmark assets](../segmentation-benchmark/README.md) first.
