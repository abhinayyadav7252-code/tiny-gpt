import sys
import types
mock_dataset = types.ModuleType('dataset')
mock_dataset.vocab_size = 50257
mock_dataset.encode = lambda x: [0]
mock_dataset.decode = lambda x: ""
sys.modules['dataset'] = mock_dataset

import torch
import config
from model import TinyGPT, print_parameter_stats

for scale in ["1.5M", "10M", "25M", "50M"]:
    config.ACTIVE_SCALE = scale
    cfg = config.SCALES[scale]
    config.embed_dim = cfg["embed_dim"]
    config.num_heads = cfg["num_heads"]
    config.num_layers = cfg["num_layers"]
    config.use_moe = cfg["use_moe"]
    config.num_experts = cfg["num_experts"]
    
    model = TinyGPT()
    print_parameter_stats(model)
