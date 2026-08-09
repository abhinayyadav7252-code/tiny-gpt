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
    text = f"User: {prompt}\nAI: {target_response}"
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
    text = f"User: {prompt}\nAI: {response}"
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

def run_alignment_training():
    print("==================================================")
    print("=== Phase 5.5: Alignment Training (SFT + DPO)  ===")
    print("==================================================\n")
    
    policy_model = TinyGPT().to(config.device)
    ref_model = TinyGPT().to(config.device)
    
    # In a real scenario, you would load the pre-trained checkpoint here
    # policy_model.load_state_dict(torch.load('checkpoints/pretrained.pt'))
    # ref_model.load_state_dict(torch.load('checkpoints/pretrained.pt'))
    
    ref_model.eval() # Reference model is always frozen
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)
    
    # Example Alignment Dataset
    dataset = [
        {
            "prompt": "What is 2+2?",
            "chosen": "The answer is 4.",
            "rejected": "I think it is 5."
        },
        {
            "prompt": "Explain gravity.",
            "chosen": "Gravity is the force that attracts a body toward the center of the earth.",
            "rejected": "Gravity is magic."
        }
    ]
    
    epochs = 1
    print(f"Training for {epochs} epochs on device: {config.device}")
    
    for epoch in range(epochs):
        total_sft_loss = 0
        total_dpo_loss = 0
        
        for item in dataset:
            optimizer.zero_grad(set_to_none=True)
            
            # Step 1: Supervised Fine-Tuning (SFT) on the chosen response
            sft_loss = compute_sft_loss(policy_model, item["prompt"], item["chosen"])
            
            # Step 2: DPO on chosen vs rejected
            dpo_loss = compute_dpo_loss(policy_model, ref_model, item["prompt"], item["chosen"], item["rejected"])
            
            # Combined Loss
            loss = sft_loss + dpo_loss
            loss.backward()
            optimizer.step()
            
            total_sft_loss += sft_loss.item()
            total_dpo_loss += dpo_loss.item()
            
        print(f"Epoch {epoch+1:02d}/{epochs} | SFT Loss: {total_sft_loss/len(dataset):.4f} | DPO Loss: {total_dpo_loss/len(dataset):.4f}", flush=True)
        
    print("\n[OK] Alignment Pipeline verified.")

if __name__ == "__main__":
    run_alignment_training()
