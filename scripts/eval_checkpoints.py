import os
import sys
import glob
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from model import TinyGPT
import json
import subprocess
import argparse
import sys

def parse_benchmark_output(output_str):
    success_rate = 0
    total = 4
    for line in output_str.split('\n'):
        if "Pipeline Success Rate:" in line:
            try:
                rate_str = line.split(":")[1].strip()
                success_rate = int(rate_str.split("/")[0].strip())
            except:
                pass
    return success_rate

def run_cognitive_benchmark(checkpoint_path):
    # We call eval_cognitive.py as a subprocess but we need a way to pass the checkpoint
    # eval_cognitive.py currently uses the hardcoded path from checkpoint/pretrain.pt
    
    backup_path = "checkpoints/pretrain.pt.bak"
    target_path = "checkpoints/pretrain.pt"
    
    if os.path.exists(target_path):
        os.rename(target_path, backup_path)
        
    try:
        import shutil
        shutil.copy2(checkpoint_path, target_path)
        
        print(f"  -> Running Regression Gate (Cognitive Benchmark)...")
        result = subprocess.run([sys.executable, "experiments/eval_cognitive.py"], capture_output=True, text=True)
        
        success_rate = parse_benchmark_output(result.stdout)
        
        return success_rate
    finally:
        if os.path.exists(target_path):
            os.remove(target_path)
        if os.path.exists(backup_path):
            os.rename(backup_path, target_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=str, default="50M")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    
    scale_dir = os.path.join(args.checkpoints_dir, args.scale)
    checkpoints = glob.glob(os.path.join(scale_dir, "*.pt"))
    checkpoints.sort(key=os.path.getmtime)
    
    if not checkpoints:
        print(f"No checkpoints found in {scale_dir}")
        return
        
    print(f"Found {len(checkpoints)} checkpoints for {args.scale} model.")
    print("="*60)
    print(f"{'Checkpoint Name':<30} | {'Step':<8} | {'Tokens':<12} | {'Cognitive Score'}")
    print("-" * 60)
    
    results = []
    
    for ckpt_path in checkpoints:
        ckpt_name = os.path.basename(ckpt_path)
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            step = ckpt.get('global_step', 'N/A')
            tokens = ckpt.get('tokens_seen', 'N/A')
            
            # Formatting tokens for readability
            if isinstance(tokens, int):
                tokens_str = f"{tokens/1e6:.2f}M"
            else:
                tokens_str = str(tokens)
                
            # Run benchmark
            score = run_cognitive_benchmark(ckpt_path)
            
            print(f"{ckpt_name:<30} | {str(step):<8} | {tokens_str:<12} | {score}/4")
            
            results.append({
                "path": ckpt_path,
                "name": ckpt_name,
                "step": step,
                "tokens": tokens,
                "score": score
            })
            
        except Exception as e:
            print(f"{ckpt_name:<30} | ERROR: {str(e)[:40]}")
            
    print("="*60)
    
    # Identify the best checkpoint (highest score)
    valid_results = [r for r in results if isinstance(r['score'], int)]
    if valid_results:
        best = max(valid_results, key=lambda x: x['score'])
        print(f"\nBest Candidate for {args.scale}: {best['name']} (Score: {best['score']}/4)")
        print(f"Path: {best['path']}")
    else:
        print("\nNo valid scores could be computed.")

if __name__ == "__main__":
    main()
