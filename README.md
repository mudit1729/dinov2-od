# DINOv2 Object Detection

This repository implements an object detection model using Facebook's DINOv2 (Vision Transformer) as the backbone feature extractor. The detector follows a DETR-like architecture with a transformer decoder on top of DINOv2 features.

## Features

- Uses DINOv2 pretrained vision transformer as the backbone feature extractor
- Implements a DETR-like decoder for object detection
- Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Supports COCO dataset format
- Includes evaluation on COCO metrics (AP, AP50, AP75, etc.)
- Supports training, validation, and test-dev evaluation

## Installation

```bash
# Clone the repository
git clone https://github.com/mudit1729/dinov2-od.git
cd dinov2-od

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage

### Training

```bash
python -m dino_detector.train \
  --train_images path/to/coco/train2017 \
  --train_annotations path/to/coco/annotations/instances_train2017.json \
  --val_images path/to/coco/val2017 \
  --val_annotations path/to/coco/annotations/instances_val2017.json \
  --output_dir outputs
```

### Resuming Training from Checkpoint

```bash
python -m dino_detector.train \
  --train_images path/to/coco/train2017 \
  --train_annotations path/to/coco/annotations/instances_train2017.json \
  --val_images path/to/coco/val2017 \
  --val_annotations path/to/coco/annotations/instances_val2017.json \
  --output_dir outputs \
  --checkpoint outputs/dino_detector_epoch_20.pth
```

### Evaluation Only

```bash
python -m dino_detector.train \
  --val_images path/to/coco/val2017 \
  --val_annotations path/to/coco/annotations/instances_val2017.json \
  --output_dir eval_outputs \
  --checkpoint outputs/dino_detector_final.pth \
  --only_evaluate
```

### Test-Dev Evaluation

```bash
python -m dino_detector.train \
  --testdev_images path/to/coco/test2017 \
  --output_dir test_outputs \
  --checkpoint outputs/dino_detector_final.pth \
  --only_evaluate
```

## Model Architecture

The model consists of three main components:

1. **DINOv2 Backbone** (frozen with LoRA adapters):
   - Uses the pre-trained DINOv2 model from Hugging Face
   - Freezes original weights for efficiency
   - Applies LoRA adapters to learn task-specific adaptations

2. **DETR-style Decoder**:
   - Transformer decoder with learned object queries
   - Multi-head self and cross-attention mechanisms
   - Processes features from the backbone to detect objects

3. **Prediction Heads**:
   - Classification head for object categories
   - Bounding box regression head using MLP

## Project Structure

```
dino_detector/
├── config.py            # Configuration parameters
├── dataset.py           # Dataset loading and processing
├── models/
│   ├── __init__.py      # Model exports
│   ├── dinov2_backbone.py # DINOv2 backbone with LoRA
│   ├── detr_decoder.py  # DETR transformer decoder
│   └── detector.py      # Full object detector model
├── train.py             # Training script
└── utils.py             # Utility functions and evaluation metrics
```

## Configuration

You can modify the model configuration in `dino_detector/config.py`:

- Model parameters (DINOv2 variant, hidden dimensions, etc.)
- Training hyperparameters (learning rate, batch size, etc.)
- LoRA parameters (rank, alpha)

## Evaluation Metrics

The model is evaluated using standard COCO evaluation metrics:

- **AP**: Average Precision at IoU=0.50:0.95
- **AP50**: Average Precision at IoU=0.50 
- **AP75**: Average Precision at IoU=0.75
- **APs**: Average Precision for small objects
- **APm**: Average Precision for medium objects
- **APl**: Average Precision for large objects

## Citation

If this code is useful for your research, please consider citing:

```
@misc{dinov2-od,
  author = {Mudit Jain},
  title = {DINOv2 Object Detection},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mudit1729/dinov2-od}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Facebook AI Research](https://ai.facebook.com/) for the DINOv2 model
- [DETR](https://github.com/facebookresearch/detr) for the transformer decoder architecture
- [LoRA](https://arxiv.org/abs/2106.09685) for the parameter-efficient fine-tuning method
- [COCO Dataset](https://cocodataset.org) for the object detection benchmark