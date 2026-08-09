import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import config
from model import TinyGPT
from dataset import encode, decode
import math

class Consolidator:
    def __init__(self, checkpoint_path, output_dir):
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.model = TinyGPT()
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=config.device))
        self.model.to(config.device)

    def prepare_data(self, replay_ratio_old=0.5):
        # 1. Load Long Term Memory (New Facts)
        memory_path = 'memory.json'
        new_text = ""
        if os.path.exists(memory_path):
            with open(memory_path, 'r') as f:
                facts = json.load(f).get("facts", [])
                for fact in facts:
                    new_text += f"{fact['key']} is {fact['value']}.\n"
        
        if not new_text:
            print("No new memories to consolidate.")
            return None
            
        # Write to a file for testing later
        os.makedirs('data', exist_ok=True)
        with open('data/memory_text.txt', 'w') as f:
            f.write(new_text)

        # 2. Load Old Data (Shakespeare Replay Buffer)
        old_path = 'data/real_text.txt'
        with open(old_path, 'r', encoding='utf-8') as f:
            old_text = f.read()

        # Calculate proportions to mix
        if replay_ratio_old == 0.0:
            final_text = new_text
        elif replay_ratio_old == 1.0:
            final_text = old_text
        else:
            new_len = len(new_text)
            old_len = len(old_text)
            target_total = max(5000, new_len + old_len)
            
            target_old = int(target_total * replay_ratio_old)
            target_new = int(target_total * (1.0 - replay_ratio_old))
            
            old_replicated = (old_text * (math.ceil(target_old / max(1, old_len))))[:target_old]
            new_replicated = (new_text * (math.ceil(target_new / max(1, new_len))))[:target_new]
            
            final_text = new_replicated + "\n" + old_replicated
            
        return encode(final_text)
        
    def sleep_cycle(self, replay_ratio_old=0.5, steps=300, save_name='step_001.pt'):
        print(f"--- Starting Sleep Cycle (Replay Ratio Old: {replay_ratio_old*100}%) ---")
        data = self.prepare_data(replay_ratio_old)
        if data is None: return
        
        data = torch.tensor(data, dtype=torch.long)
        
        def get_batch():
            ix = torch.randint(len(data) - config.context_length, (config.batch_size,))
            x = torch.stack([data[i:i+config.context_length] for i in ix])
            y = torch.stack([data[i+1:i+config.context_length+1] for i in ix])
            return x.to(config.device), y.to(config.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.model.train()
        
        for iter in range(steps):
            xb, yb = get_batch()
            logits, loss = self.model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            if iter % 50 == 0:
                print(f"Consolidation Step {iter} | Loss = {loss.item():.4f}")
                
        out_path = os.path.join(self.output_dir, save_name)
        torch.save(self.model.state_dict(), out_path)
        print(f"Consolidation Complete. Checkpoint saved to {out_path}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=float, default=0.0, help='Ratio of OLD data (0.0 means 100 percent NEW data)')
    parser.add_argument('--save', type=str, default='exp1_forgetting.pt')
    parser.add_argument('--base', type=str, default='checkpoints/phase1_baseline.pt')
    args = parser.parse_args()
    
    consolidator = Consolidator(args.base, 'checkpoints/phase3')
    consolidator.sleep_cycle(replay_ratio_old=args.ratio, steps=400, save_name=args.save)
