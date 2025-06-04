# SAM2 Fine-Tuning

This folder contains scripts and instructions for fine-tuning the **Segment Anything** (SAM2) model on custom datasets.

## Requirements

```
python>=3.8
pytorch>=1.10
opencv-python
segment-anything
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset

Place your training images and corresponding masks in `data/train` and validation data in `data/val`.

## Training

Run the fine-tuning script:

```bash
python finetune.py --data_path ./data --checkpoint /path/to/sam2_checkpoint.pth
```

## Checkpoints

The script will save fine-tuned weights in the `checkpoints/` directory.
