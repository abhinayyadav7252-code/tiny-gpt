import torch
import config
from dataset import get_batch
from model import TinyGPT
import os

print("--- Test 4: Tiny Overfit Forward/Backward Test ---")
print("Initializing Model...")

model = TinyGPT()
model = model.to(config.device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

print(f"Goal: Memorize the tiny 'hello world' dataset.")
print(f"Training on device: {config.device} for {config.max_iters} iterations.")

os.makedirs('checkpoints', exist_ok=True)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for iter in range(config.max_iters):
    if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
        losses = estimate_loss()
        print(f"Step {iter:4d} | Train Loss = {losses['train']:.4f} | Val Loss = {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\nTraining Complete! Saving checkpoint...")
torch.save(model.state_dict(), 'checkpoints/phase2_chat_tool.pt')
print("Checkpoint saved to checkpoints/phase2_chat_tool.pt")
