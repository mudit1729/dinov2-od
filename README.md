# DINOv2 Object Detection

This repository implements an object detection model using Facebook's DINOv2 (Vision Transformer) as the backbone feature extractor. The detector follows a DETR-like architecture with a transformer decoder on top of DINOv2 features.

## Features

- Uses DINOv2 pretrained vision transformer as the backbone feature extractor
- Implements a DETR-like decoder for object detection
- Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Supports COCO dataset format
- Includes evaluation on COCO metrics (AP, AP50, AP75, etc.)
- Supports training, validation, and test-dev evaluation
- Comprehensive TensorBoard logging of metrics, images, and model information
- Detailed performance monitoring and memory tracking

## Installation

```bash
# Clone the repository
git clone https://github.com/mudit1729/dinov2-od.git
cd dinov2-od

# Install dependencies
python3 -m pip install -r requirements.txt

# Install the package in development mode
python3 -m pip install -e .
```

## Usage

### Downloading COCO Dataset and Training

You can download the COCO dataset and train in a single step using the integrated download functionality:

```bash
# Download COCO training data and train
python3 -m dino_detector.train --download_train_data --output_dir outputs

# Download both training and validation data
python3 -m dino_detector.train --download_train_data --download_val_data --output_dir outputs

# Resume from checkpoint after downloading data
python3 -m dino_detector.train --download_train_data --checkpoint outputs/dino_detector_epoch_20.pth
```

For evaluation with downloaded data:

```bash
# Download validation data and evaluate a trained model
python3 -m dino_detector.train --download_val_data --only_evaluate --checkpoint outputs/dino_detector_final.pth

# Download test-dev data and evaluate
python3 -m dino_detector.train --download_test_data --only_evaluate --checkpoint outputs/dino_detector_final.pth
```

We also provide a standalone script to just download the COCO dataset:

```bash
# Just download training data
python3 download_coco.py --download_train

# Download both training and validation data
python3 download_coco.py --download_train --download_val 

# Download test data
python3 download_coco.py --download_test
```

### Training Manually

If you already have the COCO dataset downloaded, you can run training directly:

```bash
python3 -m dino_detector.train \
  --train_images path/to/coco/train2017 \
  --train_annotations path/to/coco/annotations/instances_train2017.json \
  --val_images path/to/coco/val2017 \
  --val_annotations path/to/coco/annotations/instances_val2017.json \
  --output_dir outputs
```

### Resuming Training from Checkpoint

```bash
python3 -m dino_detector.train \
  --train_images path/to/coco/train2017 \
  --train_annotations path/to/coco/annotations/instances_train2017.json \
  --val_images path/to/coco/val2017 \
  --val_annotations path/to/coco/annotations/instances_val2017.json \
  --output_dir outputs \
  --checkpoint outputs/dino_detector_epoch_20.pth
```

### Evaluation Only

```bash
python3 -m dino_detector.train \
  --val_images path/to/coco/val2017 \
  --val_annotations path/to/coco/annotations/instances_val2017.json \
  --output_dir eval_outputs \
  --checkpoint outputs/dino_detector_final.pth \
  --only_evaluate
```

### Test-Dev Evaluation

```bash
python3 -m dino_detector.train \
  --testdev_images path/to/coco/test2017 \
  --output_dir test_outputs \
  --checkpoint outputs/dino_detector_final.pth \
  --only_evaluate
```

For quick evaluations with only a small sample of test images:

```bash
# Use only 30 test images (default)
python3 -m dino_detector.train \
  --testdev_images path/to/coco/test2017 \
  --output_dir test_outputs \
  --checkpoint outputs/dino_detector_final.pth \
  --only_evaluate \
  --test_mini

# Use a custom number of test images
python3 -m dino_detector.train \
  --testdev_images path/to/coco/test2017 \
  --output_dir test_outputs \
  --checkpoint outputs/dino_detector_final.pth \
  --only_evaluate \
  --test_mini \
  --test_mini_size 50
```

### Debug/Overfit Mode

The debug/overfit mode allows training on a small subset of data to verify model convergence:

```bash
# Run training in debug mode with default settings (32 samples)
python3 -m dino_detector.train --debug --download_train_data

# Customize the number of samples and learning rate
python3 -m dino_detector.train --debug --debug_samples 64 --debug_lr 5e-4

# Use specific data with debug mode
python3 -m dino_detector.train \
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

### COCO Mini Dataset

For faster training and prototyping without going to full debug mode, you can use standardized COCO mini datasets:

```bash
# Use a 1k sample COCO mini dataset (default)
python3 -m dino_detector.train \
  --use_coco_mini \
  --train_images path/to/coco/train2017 \
  --train_annotations path/to/coco/annotations/instances_train2017.json

# Use a 5k sample COCO mini dataset
python3 -m dino_detector.train \
  --use_coco_mini \
  --coco_mini_size 5k \
  --train_images path/to/coco/train2017 \
  --train_annotations path/to/coco/annotations/instances_train2017.json

# Use COCO mini with a custom seed for different samples
python3 -m dino_detector.train \
  --use_coco_mini \
  --coco_mini_size 10k \
  --coco_mini_seed 123 \
  --train_images path/to/coco/train2017 \
  --train_annotations path/to/coco/annotations/instances_train2017.json

# Combine with lightweight mode for even faster training
python3 -m dino_detector.train \
  --use_coco_mini \
  --coco_mini_size 1k \
  --lightweight \
  --train_images path/to/coco/train2017 \
  --train_annotations path/to/coco/annotations/instances_train2017.json
```

The COCO mini datasets:
- Are standardized subsets for better reproducibility (1k, 5k, or 10k samples)
- Use standard training configuration (unlike debug mode)
- Automatically use appropriate validation set sizes
- Can be combined with other optimizations like lightweight models
- Save indices to JSON files for reproducible experiments
- Allow exact reproduction by loading from specific indices files

```bash
# Using a previously saved indices file for exact reproduction
python3 -m dino_detector.train \
  --use_coco_mini \
  --coco_mini_indices_file outputs/coco_mini_1k_seed42_indices.json \
  --train_images path/to/coco/train2017 \
  --train_annotations path/to/coco/annotations/instances_train2017.json
```

## TensorBoard Logging

The training script includes comprehensive TensorBoard logging to monitor training progress:

```bash
# Train with TensorBoard logging
python3 -m dino_detector.train \
  --train_images path/to/coco/train2017 \
  --train_annotations path/to/coco/annotations/instances_train2017.json \
  --experiment_name my_experiment

# Train with custom logging frequency and image logging
python3 -m dino_detector.train \
  --train_images path/to/coco/train2017 \
  --train_annotations path/to/coco/annotations/instances_train2017.json \
  --log_frequency 5 \
  --log_images \
  --log_images_frequency 50

# Specify a different log directory
python3 -m dino_detector.train \
  --train_images path/to/coco/train2017 \
  --train_annotations path/to/coco/annotations/instances_train2017.json \
  --log_dir logs/custom_experiment
```

To view TensorBoard logs during or after training:

```bash
tensorboard --logdir outputs/tensorboard
```

TensorBoard logging includes:
- Training and validation loss metrics
- Per-class accuracy and precision
- COCO evaluation metrics (AP, AP50, AP75, etc.)
- Learning rate and optimizer parameters
- Memory usage statistics
- Model parameter histograms and distributions
- Sample input images with bounding box visualizations (when enabled)

## Lightweight Model Configuration

For resource-constrained environments, the model can be configured with significantly fewer parameters:

```bash
# Train with lightweight configuration (uses dinov2-small by default)
python3 -m dino_detector.train \
  --train_images path/to/coco/train2017 \
  --train_annotations path/to/coco/annotations/instances_train2017.json \
  --lightweight

# Specify a different DINOv2 model variant
python3 -m dino_detector.train \
  --train_images path/to/coco/train2017 \
  --train_annotations path/to/coco/annotations/instances_train2017.json \
  --dino_model facebook/dinov2-base \
  --lightweight

# Available DINOv2 model variants: facebook/dinov2-small, facebook/dinov2-base, facebook/dinov2-large, facebook/dinov2-giant

# If you're using a checkpoint from a standard model with a lightweight model, 
# you may want to skip loading the checkpoint to avoid compatibility issues
python3 -m dino_detector.train \
  --train_images path/to/coco/train2017 \
  --train_annotations path/to/coco/annotations/instances_train2017.json \
  --checkpoint outputs/dino_detector_epoch_20.pth \
  --lightweight \
  --skip_checkpoint_load
```

The lightweight configuration:
- Uses a smaller DINOv2 variant (small by default)
- Reduces the number of decoder layers (2 instead of 6)
- Reduces the number of object queries (25 instead of 100)
- Uses smaller hidden dimensions
- Applies LoRA only to the last few transformer layers
- Uses smaller MLP for bounding box prediction
- Reduces the number of sampling points in deformable attention
- Supports all official DINOv2 model variants: small, base, large, and giant

This configuration reduces the number of trainable parameters by 80-90% with a modest trade-off in accuracy.

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
├── train.py                   # Training script with TensorBoard logging
├── utils.py                   # Utility functions, logging, and metrics
└── validate.py                # Validation and memory monitoring
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

## Analyzing Results

The repository includes a dedicated tool for analyzing detection results, generating visualizations, and creating performance reports:

```bash
# Analyze validation metrics
python3 analyze_results.py --metrics_file outputs/val_metrics_epoch_10.json

# Visualize predictions on test images
python3 analyze_results.py \
  --predictions_file outputs/testdev_predictions_final.json \
  --test_images coco_data/test2017 \
  --num_visualizations 10 \
  --confidence_threshold 0.6

# Run evaluation and generate analysis in one step
python3 analyze_results.py \
  --run_eval \
  --model_path outputs/dino_detector_final.pth \
  --val_images path/to/coco/val2017 \
  --val_annotations path/to/coco/annotations/instances_val2017.json \
  --output_dir analysis_outputs
```

The analyze_results.py script generates:
- Detailed metrics tables and charts
- Confidence score distributions
- Class distribution analysis
- Sample visualizations with bounding boxes
- Summary reports for model performance

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