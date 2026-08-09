import torch
import torch.nn.functional as F
import config
from model import TinyGPT
from dataset import encode

def compute_sft_loss(model, prompt, target_response):
    """
    Supervised Fine-Tuning Loss
    Trains the model to generate the target_response given the prompt.
    """
    text = f"User: {prompt}\nAI: {target_response}<|endoftext|>"
    idx = torch.tensor(encode(text), dtype=torch.long, device=config.device).unsqueeze(0)
    
    # We only want to compute loss on the AI response part
    prompt_len = len(encode(f"User: {prompt}\nAI: "))
    
    # targets is the same as idx but shifted by 1
    inputs = idx[:, :-1]
    targets = idx[:, 1:].clone()
    
    # Ignore loss on the prompt tokens (set target to -100)
    targets[:, :prompt_len-1] = -100
    
    logits, loss, _ = model(inputs, targets=targets)
    return loss

def compute_dpo_loss(policy_model, ref_model, prompt, chosen, rejected, beta=0.1):
    """
    Direct Preference Optimization (DPO) Loss
    """
    # 1. Get log probs from policy model (the one being trained)
    policy_chosen_logprob = get_response_logprob(policy_model, prompt, chosen)
    policy_rejected_logprob = get_response_logprob(policy_model, prompt, rejected)
    
    # 2. Get log probs from reference model (frozen original model)
    with torch.no_grad():
        ref_chosen_logprob = get_response_logprob(ref_model, prompt, chosen)
        ref_rejected_logprob = get_response_logprob(ref_model, prompt, rejected)
        
    # 3. Compute DPO loss
    chosen_ratio = policy_chosen_logprob - ref_chosen_logprob
    rejected_ratio = policy_rejected_logprob - ref_rejected_logprob
    
    loss = -F.logsigmoid(beta * (chosen_ratio - rejected_ratio))
    return loss.mean()

def get_response_logprob(model, prompt, response):
    text = f"User: {prompt}\nAI: {response}<|endoftext|>"
    idx = torch.tensor(encode(text), dtype=torch.long, device=config.device).unsqueeze(0)
    prompt_len = len(encode(f"User: {prompt}\nAI: "))
    
    inputs = idx[:, :-1]
    targets = idx[:, 1:]
    
    logits, _, _ = model(inputs)
    
    # Calculate log probs for the actual target tokens
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Gather the log probs corresponding to the target tokens
    gathered_log_probs = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)
    
    # Only sum the log probs of the response, not the prompt
    response_log_probs = gathered_log_probs[:, prompt_len-1:]
    return response_log_probs.sum(dim=1)

import json
import os
import argparse

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def run_sft_training(dataset_path, epochs=1, save_path="checkpoints/sft_model.pt", lr=1e-4):
    print("==================================================")
    print(f"=== Phase 6.3: SFT Training ({dataset_path}) ===")
    print("==================================================\n")
    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
        
    model = TinyGPT().to(config.device)
    # We start from scratch (untrained) to prove the pipeline works, 
    # but in a real scenario we'd load the pre-trained weights.
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataset = load_jsonl(dataset_path)
    
    print(f"Training on {len(dataset)} examples for {epochs} epochs on device: {config.device}")
    
    for epoch in range(epochs):
        total_sft_loss = 0
        
        for item in dataset:
            optimizer.zero_grad(set_to_none=True)
            
            # The JSONL might have "prompt" and "completion", or just "text"
            if "prompt" in item and "completion" in item:
                # Remove "User: " and "AI: " if they are already in the dataset prompt to avoid double-adding
                prompt = item["prompt"]
                if prompt.startswith("User: "): prompt = prompt[6:]
                if prompt.endswith("\nAI:"): prompt = prompt[:-4]
                if prompt.endswith("\nAI: "): prompt = prompt[:-5]
                
                loss = compute_sft_loss(model, prompt, item["completion"])
            else:
                # For pure text completion (pretrain), no masking
                idx = torch.tensor(encode(item["text"]), dtype=torch.long, device=config.device).unsqueeze(0)
                inputs = idx[:, :-1]
                targets = idx[:, 1:].clone()
                _, loss, _ = model(inputs, targets=targets)
            
            loss.backward()
            optimizer.step()
            total_sft_loss += loss.item()
            
        avg_loss = total_sft_loss / len(dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:03d}/{epochs} | SFT Loss: {avg_loss:.4f}", flush=True)
            
    torch.save(model.state_dict(), save_path)
    print(f"\n[OK] Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["overfit", "full"], default="overfit")
    args = parser.parse_args()
    
    if args.mode == "overfit":
        run_sft_training("data/sft_overfit_data.jsonl", epochs=30, save_path="checkpoints/sft_overfit.pt", lr=5e-4)
    elif args.mode == "full":
        run_sft_training("data/mixed_training_data.jsonl", epochs=5, save_path="checkpoints/sft_full.pt", lr=1e-4)
