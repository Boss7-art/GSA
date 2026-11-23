# Semantic segmentation benchmark

The dataset contains 34 photographs from five shopping centres with 1,020 manually annotated shapes in classes A–H. EMSANet uses all 34 photographs. PGDENet uses 24 training photographs and 10 held-out photographs. `results.csv` reports dataset-level Hungarian-matched mean IoU over labelled pixels.

Extract `segmentation-benchmark-assets.zip` at the **repository root**. The archive supplies the photographs, 10 validation depth maps and the fine-tuned PGDENet parameters. Its SHA-256 is `210c5010fe6b7ee9a60b45ffb7678364d710a61a396b9c13b0f93ffa05588ef9`.

Run these commands from this directory with Python 3.12.

```text
python -m pip install -r requirements.txt
python prepare_models.py
python reproduce.py
```

Use `python reproduce.py --model pgdenet` or `--model emsanet` to evaluate one model. Outputs are written to `results/`. The PGDENet inference pass uses the validation depth maps supplied with the photographs. Training uses inverted depth on the training split.

To fit PGDENet again after generating depth maps with the complete workflow, run:

```text
python tools/finetune_pgdenet_ah_semantic.py --prepared-root results --depth-dir results/emsanet_samples/depth_anything_v2_small_sunrgbd_stats_inverted --train-list data/splits/train.txt --eval-list data/splits/validation.txt --epochs 16 --learning-rate 0.00005 --class-weight-power 1 --max-class-weight 50 --dice-weight 1 --aux-weight 0.15 --seed 17 --checkpoint-out results/pgdenet.safetensors --output-subdir pgdenet_refit
```

| Resource | Source |
|---|---|
| EMSANet, pretrained on SUN RGB-D | [Seichter et al.](https://doi.org/10.1109/IJCNN55064.2022.9892852), [code and weights](https://github.com/TUI-NICR/EMSANet#license-and-citations) |
| NICR model and dataset libraries | [Model library](https://github.com/TUI-NICR/nicr-multitask-scene-analysis), [dataset library](https://github.com/TUI-NICR/nicr-scene-analysis-datasets) |
| PGDENet | [Yang et al., source implementation](https://github.com/EnquanYang2022/PGDENet) |
| Depth Anything V2 Small | [Model card](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf), [Apache 2.0 license](https://github.com/DepthAnything/Depth-Anything-V2#license) |
