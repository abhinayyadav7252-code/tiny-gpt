import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import config
from brain.consolidate import Consolidator
from experiments.forgetting_test import evaluate_loss

print("Calculating Baseline Losses (Before Sleep)...")
model = Consolidator('checkpoints/phase1_baseline.pt', 'checkpoints/phase3').model
with open('data/real_text.txt', 'r', encoding='utf-8') as f:
    old_text = f.read()
with open('data/memory_text.txt', 'r', encoding='utf-8') as f:
    new_text = f.read()

# Use 100 iters for a very stable evaluation
L_old_before = evaluate_loss(model, old_text, iters=100)
L_new_before = evaluate_loss(model, new_text, iters=100)

print(f"Baseline L_old_before: {L_old_before:.4f}")
print(f"Baseline L_new_before: {L_new_before:.4f}\n")

# ratios map to Replay Ratio Old (e.g., 0.0 means 0% old, 100% new)
ratios = [0.0, 0.25, 0.50, 0.75]
results = []

for ratio_old in ratios:
    ratio_new = 1.0 - ratio_old
    print(f"\n=== Running Sweep for Ratio Old: {ratio_old*100}%, Ratio New: {ratio_new*100}% ===")
    
    # Always start fresh from Phase 1 baseline
    consolidator = Consolidator('checkpoints/phase1_baseline.pt', 'checkpoints/phase3')
    save_name = f"exp3_ratio_{int(ratio_old*100)}.pt"
    
    # Run sleep cycle (fine-tune)
    consolidator.sleep_cycle(replay_ratio_old=ratio_old, steps=300, save_name=save_name)
    
    # Evaluate
    # Load the newly trained model for evaluation
    eval_model = Consolidator(os.path.join('checkpoints/phase3', save_name), 'checkpoints/phase3').model
    
    L_old_after = evaluate_loss(eval_model, old_text, iters=100)
    L_new_after = evaluate_loss(eval_model, new_text, iters=100)
    
    F = L_old_after - L_old_before
    delta_new = L_new_before - L_new_after
    
    results.append({
        'ratio_new': ratio_new,
        'ratio_old': ratio_old,
        'L_old_after': L_old_after,
        'L_new_after': L_new_after,
        'Forgetting (F)': F,
        'Learning (dNew)': delta_new
    })

print("\n" + "="*80)
print("Experiment 3: Replay Ratio Sweep Results")
print("="*80)
print(f"{'Ratio (New/Old)':<15} | {'L_old_after':<12} | {'L_new_after':<12} | {'Forgetting (F)':<15} | {'Learning (dNew)':<15}")
print("-" * 80)
for r in results:
    ratio_str = f"{int(r['ratio_new']*100)}/{int(r['ratio_old']*100)}"
    print(f"{ratio_str:<15} | {r['L_old_after']:<12.4f} | {r['L_new_after']:<12.4f} | {r['Forgetting (F)']:<15.4f} | {r['Learning (dNew)']:<15.4f}")
print("="*80)
