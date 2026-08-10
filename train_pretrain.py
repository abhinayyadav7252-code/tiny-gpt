import torch
import numpy as np
import config
from model import TinyGPT, print_parameter_stats
import argparse
import os
import math
from tqdm import tqdm

def get_batch(data, batch_size, context_length, device):
    ix = torch.randint(len(data) - context_length, (batch_size,))
    
    # We loaded data as uint16, convert to int64 for torch.long
    x = torch.stack([torch.from_numpy((data[i:i+context_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+context_length]).astype(np.int64)) for i in ix])
    
    return x.to(device), y.to(device)

def estimate_loss(model, train_data, val_data, eval_iters):
    model.eval()
    out = {}
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, config.batch_size, config.context_length, config.device)
            with torch.no_grad():
                logits, loss, _ = model(X, Y)
                losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=str, choices=list(config.SCALES.keys()), default="10M")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--save_path", type=str, default="checkpoints/pretrain.pt")
    args = parser.parse_args()
    
    config.ACTIVE_SCALE = args.scale
    cfg = config.SCALES[args.scale]
    config.embed_dim = cfg["embed_dim"]
    config.num_heads = cfg["num_heads"]
    config.num_layers = cfg["num_layers"]
    config.use_moe = cfg["use_moe"]
    config.num_experts = cfg["num_experts"]
    
    # Load dataset
    print("Loading binary dataset...")
    train_path = os.path.join(args.data_dir, "train.bin")
    val_path = os.path.join(args.data_dir, "val.bin")
    
    # memory map for speed
    train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
    print(f"Loaded {len(train_data):,} training tokens")
    
    # Initialize model
    model = TinyGPT().to(config.device)
    print_parameter_stats(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Number of iterations is roughly total tokens / (batch_size * context_length)
    tokens_per_iter = config.batch_size * config.context_length
    max_iters = (len(train_data) // tokens_per_iter) * args.epochs
    
    print(f"Starting Pretraining for {max_iters} iterations ({args.epochs} epochs)...")
    
    pbar = tqdm(range(max_iters))
    for iter_num in pbar:
        
        # Periodic evaluation
        if iter_num % config.eval_interval == 0 or iter_num == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, config.eval_iters)
            val_ppl = math.exp(losses['val'])
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val PPL {val_ppl:.2f}")
            pbar.set_description(f"Loss: {losses['train']:.4f}")
            
        xb, yb = get_batch(train_data, config.batch_size, config.context_length, config.device)
        
        logits, loss, aux_loss = model(xb, yb)
        
        if config.use_moe:
            loss = loss + config.moe_loss_coef * aux_loss
            
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved pretrained checkpoint to {args.save_path}")

if __name__ == "__main__":
    main()
