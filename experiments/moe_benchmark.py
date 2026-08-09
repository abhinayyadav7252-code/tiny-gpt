import sys; import os; sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import config
from model import TinyGPT
from dataset import get_batch
import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_eval(name):
    print(f"\n{'='*50}\n=== Training {name} ===\n{'='*50}")
    model = TinyGPT().to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    print(f"Total Parameters: {count_parameters(model):,}")
    
    # Train for a very short duration just to observe behavior
    iters = 5
    start_time = time.time()
    
    for iter in range(iters):
        xb, yb = get_batch('train')
        logits, loss, _ = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        print(f"Step {iter:02d} | Loss: {loss.item():.4f}", flush=True)
            
    end_time = time.time()
    print(f"Training Time ({iters} steps): {end_time - start_time:.2f}s")
    return model

if __name__ == '__main__':
    # 1. Train Dense Model
    config.use_moe = False
    dense_model = train_and_eval("Dense Model")
    
    # 2. Train MoE Model
    # Reduce embed_dim to keep parameter count roughly equal for fair comparison
    # In MoE, each block has 4 experts instead of 1 FFN. 
    # To keep size similar to Dense(embed_dim=512), we scale embed_dim down slightly.
    config.embed_dim = 384 
    config.num_heads = 6
    config.use_moe = True
    config.num_experts = 4
    config.top_k_experts = 2
    
    moe_model = train_and_eval("Sparse MoE Model (4 Experts, Top-2 Routing)")
    
    print("\n[Conclusion]")
    print("MoE introduces sparsity, meaning compute per token is lower even if total parameters are high.")
    print("Check parameter counts and step times above to verify efficiency.")
