# Hyperparameters for Project Chaitanya Phase 5 (Frontier Scale)
batch_size = 4
context_length = 512
embed_dim = 512
num_heads = 8
num_layers = 12
dropout = 0.1

# MoE Settings
use_moe = False
num_experts = 4
top_k_experts = 2
moe_loss_coef = 0.1

# Training settings
learning_rate = 3e-4
max_iters = 500
eval_interval = 50
eval_iters = 10

# Dynamic device (Colab ready)
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Target: 50-80M Parameters (approx 160MB in FP32, 80MB in FP16)
