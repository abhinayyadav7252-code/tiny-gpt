import torch
import config
from model import TinyGPT
from dataset import encode, decode, vocab_size
import os

print("--- Test A: Random Initialization Generation ---")
raw_model = TinyGPT().to(config.device)
start_char = 'F'
context = torch.tensor(encode(start_char), dtype=torch.long, device=config.device).unsqueeze(0)

print(f"Starting prompt: '{start_char}'")
raw_idx = raw_model.generate(context, max_new_tokens=50)[0].tolist()
print(decode(raw_idx))
print("-" * 40)

print("\n--- Test C: Real Corpus Generation ---")
checkpoint_path = 'checkpoints/tiny_gpt_real.pt'

if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint {checkpoint_path} not found. Please run train.py first.")
    exit(1)

model = TinyGPT()
model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
model.eval()
model.to(config.device)

trained_idx = model.generate(context, max_new_tokens=100)[0].tolist()
print(decode(trained_idx))
print("-" * 40)
