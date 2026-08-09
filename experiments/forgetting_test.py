import torch
import sys
import os

# Add parent directory to path so we can import model and config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from model import TinyGPT
from dataset import encode, decode
import argparse

def evaluate_loss(model, text, iters=50):
    data = torch.tensor(encode(text), dtype=torch.long)
    if len(data) <= config.context_length:
        # Replicate to make it long enough to evaluate
        data = data.repeat((config.context_length // len(data)) + 2)
        
    def get_batch():
        ix = torch.randint(len(data) - config.context_length, (config.batch_size,))
        x = torch.stack([data[i:i+config.context_length] for i in ix])
        y = torch.stack([data[i+1:i+config.context_length+1] for i in ix])
        return x.to(config.device), y.to(config.device)

    model.eval()
    losses = torch.zeros(iters)
    with torch.no_grad():
        for k in range(iters):
            X, Y = get_batch()
            logits, loss = model(X, Y)
            losses[k] = loss.item()
    return losses.mean().item()

def measure_retention(model, questions):
    print("\n--- Generation Retention Test ---")
    model.eval()
    for q in questions:
        prompt = q
        context = torch.tensor(encode(prompt), dtype=torch.long, device=config.device).unsqueeze(0)
        idx = model.generate(context, max_new_tokens=40)[0].tolist()
        print(f"Q: {q}")
        print(f"A: {decode(idx)[len(prompt):].strip()}")
        print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint to test')
    args = parser.parse_args()
    
    print(f"\n=============================================")
    print(f"Evaluating Checkpoint: {args.checkpoint}")
    print(f"=============================================")
    
    # Load model
    model = TinyGPT()
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=config.device))
    model.to(config.device)
    
    # Load old and new data
    with open('data/real_text.txt', 'r', encoding='utf-8') as f:
        old_text = f.read()
        
    try:
        with open('data/memory_text.txt', 'r', encoding='utf-8') as f:
            new_text = f.read()
    except FileNotFoundError:
        print("data/memory_text.txt not found. Did you run consolidate.py first?")
        exit(1)
        
    l_old = evaluate_loss(model, old_text)
    l_new = evaluate_loss(model, new_text)
    
    print(f"L_old (Shakespeare Loss): {l_old:.4f}")
    print(f"L_new (Memory Loss)     : {l_new:.4f}")
    
    measure_retention(model, [
        "Abhinav is ", # Direct
        "User's name is ", # Direct
        "Who is learning Python?", # Paraphrased
        "What is the capital of France?" # Control
    ])
