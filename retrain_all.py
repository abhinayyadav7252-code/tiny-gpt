import torch
import config
from model import TinyGPT
import dataset
import os

os.makedirs('checkpoints/phase3', exist_ok=True)

def train_model(data_file, save_path, iters=500):
    dataset.data_path = data_file
    model = TinyGPT().to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for i in range(iters):
        xb, yb = dataset.get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), save_path)
    print(f"Saved {save_path}")

print("Retraining Phase 1 Baseline (Shakespeare)...")
train_model('data/real_text.txt', 'checkpoints/phase1_baseline.pt', 500)

print("Retraining Phase 2 Baseline (Chat/Math)...")
train_model('data/chat_data.txt', 'checkpoints/phase2_baseline.pt', 500)

print("Done retraining baselines with universal tokenizer!")
