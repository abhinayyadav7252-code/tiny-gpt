import torch

# Define architecture scaling parameters
SCALES = {
    "1.5M": {"embed_dim": 64, "num_heads": 4, "num_layers": 4, "num_experts": 4, "use_moe": False},
    "10M":  {"embed_dim": 128, "num_heads": 4, "num_layers": 4, "num_experts": 4, "use_moe": True},
    "25M":  {"embed_dim": 256, "num_heads": 4, "num_layers": 6, "num_experts": 4, "use_moe": True},
    "50M":  {"embed_dim": 384, "num_heads": 6, "num_layers": 6, "num_experts": 4, "use_moe": True}
}

ACTIVE_SCALE = "10M"
cfg = SCALES[ACTIVE_SCALE]

# Global Hyperparameters
batch_size = 4
context_length = 512
embed_dim = cfg["embed_dim"]
num_heads = cfg["num_heads"]
num_layers = cfg["num_layers"]
dropout = 0.1

# MoE Settings
use_moe = cfg["use_moe"]
num_experts = cfg["num_experts"]
top_k_experts = 2
moe_loss_coef = 0.1

# Training settings
learning_rate = 3e-4
max_iters = 500
eval_interval = 50
eval_iters = 10

# Dynamic device (Colab ready)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
