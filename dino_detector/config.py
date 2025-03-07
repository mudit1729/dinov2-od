# config.py
# Configuration and hyperparameters

# General training parameters
learning_rate = 1e-4
num_epochs = 50
batch_size = 8  # Batch size per GPU
num_workers = 4  # Number of data loading workers per GPU

# Distributed training parameters
distributed_backend = "nccl"  # Backend for distributed training (nccl for GPU, gloo for CPU)
find_unused_parameters = True  # For DDP, find unused parameters in the model

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
gradient_accumulation_steps = 1  # For larger effective batch sizes with limited GPU memory
gradient_clip_val = 1.0  # For gradient clipping to prevent exploding gradients