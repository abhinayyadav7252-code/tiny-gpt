import torch
import numpy as np
import config
from model import TinyGPT, print_parameter_stats
import argparse
import os
import math
import glob
from tqdm import tqdm
import random

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

def cleanup_old_checkpoints(save_dir, keep_last=3):
    checkpoints = glob.glob(os.path.join(save_dir, "ckpt_step_*.pt"))
    checkpoints.sort(key=os.path.getmtime)
    while len(checkpoints) > keep_last:
        oldest = checkpoints.pop(0)
        os.remove(oldest)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=str, choices=list(config.SCALES.keys()), default="10M")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    config.ACTIVE_SCALE = args.scale
    cfg = config.SCALES[args.scale]
    config.embed_dim = cfg["embed_dim"]
    config.num_heads = cfg["num_heads"]
    config.num_layers = cfg["num_layers"]
    config.use_moe = cfg["use_moe"]
    config.num_experts = cfg["num_experts"]
    
    scale_save_dir = os.path.join(args.save_dir, args.scale)
    os.makedirs(scale_save_dir, exist_ok=True)
    
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    start_iter = 0
    tokens_seen = 0
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        start_iter = checkpoint.get('global_step', 0)
        tokens_seen = checkpoint.get('tokens_seen', 0)
        
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'])
        if 'numpy_rng_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_rng_state'])
        if 'random_rng_state' in checkpoint:
            random.setstate(checkpoint['random_rng_state'])
            
        print(f"Resumed at step {start_iter} | tokens seen: {tokens_seen:,}")
    else:
        print_parameter_stats(model)
    
    tokens_per_iter = config.batch_size * config.context_length
    max_iters = (len(train_data) // tokens_per_iter) * args.epochs
    
    if start_iter >= max_iters:
        print("Training already completed.")
        return
        
    print(f"Starting Pretraining from step {start_iter} to {max_iters}...")
    
    pbar = tqdm(range(start_iter, max_iters), initial=start_iter, total=max_iters)
    for iter_num in pbar:
        
        # Periodic evaluation and saving
        if (iter_num > start_iter and iter_num % args.save_interval == 0) or iter_num == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, config.eval_iters)
            val_ppl = math.exp(losses['val'])
            print(f"\n[Step {iter_num}] train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val PPL {val_ppl:.2f}")
            
            # Save checkpoint
            ckpt_path = os.path.join(scale_save_dir, f"ckpt_step_{iter_num}.pt")
            checkpoint = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'global_step': iter_num,
                'tokens_seen': tokens_seen,
                'config': {
                    'scale': args.scale,
                    'embed_dim': config.embed_dim,
                    'num_heads': config.num_heads,
                    'num_layers': config.num_layers,
                },
                'torch_rng_state': torch.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'random_rng_state': random.getstate()
            }
            torch.save(checkpoint, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")
            cleanup_old_checkpoints(scale_save_dir, keep_last=3)
            
        xb, yb = get_batch(train_data, config.batch_size, config.context_length, config.device)
        
        logits, loss, _ = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        tokens_seen += tokens_per_iter
        
        if iter_num % 10 == 0:
            pbar.set_description(f"Loss: {loss.item():.4f}")
            
    final_save_path = os.path.join(scale_save_dir, "final.pt")
    torch.save({'model_state': model.state_dict(), 'config': cfg}, final_save_path)
    print(f"Saved final checkpoint to {final_save_path}")

if __name__ == "__main__":
    main()
