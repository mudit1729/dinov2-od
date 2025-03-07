# DINOv2 Object Detector

This project implements an object detection system using a frozen DINOv2 vision transformer backbone with LoRA adapters and a DETR-style transformer decoder with learned object queries.

## 📋 Features

- **Efficient Fine-tuning**: Uses LoRA (Low-Rank Adaptation) to fine-tune the DINOv2 backbone efficiently with minimal trainable parameters
- **DETR Architecture**: Implements a transformer decoder with object queries for end-to-end object detection
- **Modular Design**: Separates backbone, decoder, and detection components for easy experimentation
- **Customizable**: Easy to adjust hyperparameters like LoRA rank, decoder depth, and learning rate

## 🧠 Architecture

The model architecture consists of three main components:

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

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dinov2-od.git
cd dinov2-od

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Training

To train the model on your dataset:

```bash
python -m dino_detector.train
```

You can adjust training parameters in `dino_detector/config.py`.

### Customizing the Model

To experiment with different hyperparameters, modify `dino_detector/config.py`:

```python
# Change LoRA rank
lora_r = 8  # Default is 4

# Adjust decoder depth
num_decoder_layers = 4  # Default is 6

# Change learning rate
learning_rate = 5e-5  # Default is 1e-4
```

## 📚 Project Structure

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
└── utils.py             # Utility functions including LoRA implementation
```

## 📊 Research Notes

This implementation is designed for research experimentation. Some potential research directions:

- Compare different LoRA ranks and their effect on performance
- Experiment with the number of object queries and decoder layers
- Analyze the effect of different learning rates on convergence
- Compare performance with and without LoRA adapters

## 📝 Citation

If this code is useful for your research, please consider citing:

```
@misc{dinov2-od,
  author = {Your Name},
  title = {DINOv2 Object Detector},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/dinov2-od}}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Facebook AI Research](https://ai.facebook.com/) for the DINOv2 model
- [DETR](https://github.com/facebookresearch/detr) for the transformer decoder architecture
- [LoRA](https://arxiv.org/abs/2106.09685) for the parameter-efficient fine-tuning method