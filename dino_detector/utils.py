# utils.py
import torch.nn as nn

class MLP(nn.Module):
    """
    Very simple multi-layer perceptron (used for bounding box prediction).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


def add_lora_to_module(module, r=4, alpha=1.0):
    """
    Recursively replace nn.Linear layers with a LoRA-adapted version.
    Only replace layers that are instances of nn.Linear.
    """
    for name, child in module.named_children():
        # Recursively apply on child modules first
        add_lora_to_module(child, r=r, alpha=alpha)
        if isinstance(child, nn.Linear):
            # Replace with a LoRA-adapted linear layer
            setattr(module, name, LoraLinear(child, r=r, alpha=alpha))


class LoraLinear(nn.Module):
    """
    LoRA adapter applied to a linear layer.
    It freezes the original weight and learns a low-rank update.
    """
    def __init__(self, linear_layer: nn.Linear, r=4, alpha=1.0):
        super().__init__()
        self.linear = linear_layer
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.r = r
        self.alpha = alpha

        # Create low-rank matrices A and B
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        # Initialize lora_B with zeros so initially only original weight is used
        nn.init.zeros_(self.lora_B.weight)
        # Freeze original parameters
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Original output plus scaled low-rank update
        return self.linear(x) + self.alpha * self.lora_B(self.lora_A(x))