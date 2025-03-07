# config.py
# Configuration and hyperparameters

# General training parameters
learning_rate = 1e-4
num_epochs = 50
batch_size = 8
num_workers = 4

# Model parameters
dino_model_name = "facebook/dinov2-base"  # Pretrained DINOv2 from Hugging Face
lora_r = 4          # Rank for LoRA adapter
lora_alpha = 1.0    # Scaling factor for LoRA updates
hidden_dim = 768    # Hidden dimension (should match the DINOv2 model)
num_queries = 100   # Number of learned object queries for the decoder
num_decoder_layers = 6
nheads = 8
num_classes = 91    # For example, COCO has 91 classes

# Training and optimizer settings
weight_decay = 1e-4