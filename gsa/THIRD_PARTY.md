# Third-party models

| Model | Weight location | Upstream source and licence |
|---|---|---|
| SegFormer B0, ADE20K | `segformer_b0/` | [NVIDIA](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) / [SegFormer licence](licenses/NVIDIA-SegFormer.txt) |
| Grounding DINO Tiny | `dino_tiny/` | [IDEA Research](https://huggingface.co/IDEA-Research/grounding-dino-tiny) / [Apache 2.0 and notices](licenses/GroundingDINO.txt) |
| Grounding DINO Swin-T OGC | `detector.safetensors`, `bert-base-uncased/model.safetensors` | [IDEA Research](https://github.com/IDEA-Research/GroundingDINO#checkpoints) / [Apache 2.0 and notices](licenses/GroundingDINO.txt) |
| BERT Base Uncased tokenizer and configuration | `bert-base-uncased/` | [Google Research](https://huggingface.co/google-bert/bert-base-uncased) / [Apache 2.0](LICENSE) |
| Segment Anything ViT-H | `sam_vit_h_4b8939.pth` | [Meta](https://github.com/facebookresearch/segment-anything#license) / [Apache 2.0](LICENSE) |
| SAM 2.1 Hiera Small | `sam21_small/` | [Meta](https://huggingface.co/facebook/sam2.1-hiera-small) / [Apache 2.0](LICENSE) |
| Qwen3-VL-2B-Instruct | `qwen3_vl_2b/` | [Qwen](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) / [Apache 2.0](LICENSE) |
| MoGe-2 ViT-S Normal | `moge/model.pt` | [Model weights (MIT)](https://huggingface.co/Ruicheng/moge-2-vits-normal) / [MoGe and DINOv2 notices](licenses/MoGe.txt) |

Weight locations are relative to the weights directory. SegFormer is licensed for noncommercial research or evaluation. Redistribution requires the original notices and a copy of the licence. [SegFormer licence](licenses/NVIDIA-SegFormer.txt)
