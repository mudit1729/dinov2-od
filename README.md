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

### Downloading COCO Dataset and Training

You can download the COCO dataset and train in a single step using the integrated download functionality:

```bash
# Download COCO training data and train
python -m dino_detector.train --download_train_data --output_dir outputs

# Download both training and validation data
python -m dino_detector.train --download_train_data --download_val_data --output_dir outputs

# Resume from checkpoint after downloading data
python -m dino_detector.train --download_train_data --checkpoint outputs/dino_detector_epoch_20.pth
```

For evaluation with downloaded data:

```bash
# Download validation data and evaluate a trained model
python -m dino_detector.train --download_val_data --only_evaluate --checkpoint outputs/dino_detector_final.pth

# Download test-dev data and evaluate
python -m dino_detector.train --download_test_data --only_evaluate --checkpoint outputs/dino_detector_final.pth
```

We also provide a standalone script to just download the COCO dataset:

```bash
# Just download training data
python download_coco.py --download_train

# Download both training and validation data
python download_coco.py --download_train --download_val 

# Download test data
python download_coco.py --download_test
```

### Training Manually

If you already have the COCO dataset downloaded, you can run training directly:

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

### Debug/Overfit Mode

The debug/overfit mode allows training on a small subset of data to verify model convergence:

```bash
# Run training in debug mode with default settings (32 samples)
python -m dino_detector.train --debug --download_train_data

# Customize the number of samples and learning rate
python -m dino_detector.train --debug --debug_samples 64 --debug_lr 5e-4

# Use specific data with debug mode
python -m dino_detector.train \
  --debug \
  --train_images path/to/coco/train2017 \
  --train_annotations path/to/coco/annotations/instances_train2017.json
```

The debug mode:
- Uses a small subset of data (32 samples by default)
- Runs for more epochs (100 by default instead of 50)
- Uses a higher learning rate for faster convergence
- Validates more frequently to monitor progress

This mode is useful for:
- Verifying that the model architecture is implemented correctly
- Ensuring the loss functions are properly configured
- Testing that the model can overfit before training on full data
- Debugging gradient flow issues

## Model Architecture

The model consists of three main components:

1. **DINOv2 Backbone** (frozen with LoRA adapters):
   - Uses the pre-trained DINOv2 model from Hugging Face
   - Freezes original weights for efficiency
   - Applies LoRA adapters to learn task-specific adaptations

2. **Transformer Decoder**:
   - Supports both standard and deformable attention mechanisms
   - Learned object queries for end-to-end detection
   - Deformable attention for better convergence on complex scenes
   - Multi-head self and cross-attention mechanisms

3. **Prediction Heads**:
   - Classification head for object categories
   - Bounding box regression head using MLP

## Project Structure

```
dino_detector/
├── config.py                  # Configuration parameters
├── dataset.py                 # Dataset loading and processing
├── losses.py                  # Loss functions with Hungarian matching
├── matching.py                # Bipartite matching for predictions to GT
├── models/
│   ├── __init__.py            # Model exports
│   ├── dinov2_backbone.py     # DINOv2 backbone with LoRA
│   ├── detr_decoder.py        # Transformer decoder
│   ├── deformable_attention.py# Deformable attention modules
│   └── detector.py            # Full object detector model
├── train.py                   # Training script
└── utils.py                   # Utility functions and evaluation metrics
```

## Configuration

You can modify the model configuration in `dino_detector/config.py`:

- Model parameters (DINOv2 variant, hidden dimensions, etc.)
- Training hyperparameters (learning rate, batch size, etc.)
- LoRA parameters (rank, alpha)
- Deformable attention parameters (sampling points, modulation)
- Debug mode settings (subset size, learning rate)
- Loss function weights for Hungarian matching

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