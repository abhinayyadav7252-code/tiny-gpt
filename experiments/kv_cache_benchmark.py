import sys; import os; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import config
from model import TinyGPT
from dataset import encode, decode

def benchmark_generation(model, prompt, max_tokens=100, use_cache=True):
    context = torch.tensor(encode(prompt), dtype=torch.long, device=config.device).unsqueeze(0)
    
    # Warmup
    _ = model.generate(context, max_new_tokens=10, use_cache=use_cache)
    
    # Benchmark
    torch.cuda.synchronize() if config.device == 'cuda' else None
    start_time = time.time()
    
    _ = model.generate(context, max_new_tokens=max_tokens, use_cache=use_cache)
    
    torch.cuda.synchronize() if config.device == 'cuda' else None
    end_time = time.time()
    
    time_taken = end_time - start_time
    tokens_per_sec = max_tokens / time_taken
    return time_taken, tokens_per_sec

if __name__ == '__main__':
    print(f"--- Phase 5.2 KV-Cache Benchmark ---")
    print(f"Device: {config.device}")
    model = TinyGPT().to(config.device)
    model.eval()
    
    prompt = "The quick brown fox"
    max_tokens = 50
    
    print("\nBenchmarking WITHOUT KV Cache...")
    time_no_cache, tps_no_cache = benchmark_generation(model, prompt, max_tokens, use_cache=False)
    print(f"Time: {time_no_cache:.2f}s | Speed: {tps_no_cache:.2f} tokens/sec")
    
    print("\nBenchmarking WITH KV Cache...")
    time_cache, tps_cache = benchmark_generation(model, prompt, max_tokens, use_cache=True)
    print(f"Time: {time_cache:.2f}s | Speed: {tps_cache:.2f} tokens/sec")
    
    speedup = tps_cache / tps_no_cache
    print(f"\nResult: KV Cache provided a {speedup:.2f}x speedup.")
