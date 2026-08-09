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

L_old_before = evaluate_loss(model, old_text, iters=100)
L_new_before = evaluate_loss(model, new_text, iters=100)

print(f"Baseline L_old_before: {L_old_before:.4f}")
print(f"Baseline L_new_before: {L_new_before:.4f}\n")

# We use 75% New / 25% Old for this test to see if the slight forgetting accumulates.
ratio_old = 0.25
num_cycles = 5
results = []
current_checkpoint = 'checkpoints/phase1_baseline.pt'

print("=== Running Experiment 4: Repeated Sleep Cycles ===")
print(f"Using Ratio Old: {ratio_old*100}%, Ratio New: {(1.0-ratio_old)*100}%\n")

for cycle in range(1, num_cycles + 1):
    print(f"\n--- Cycle {cycle} ---")
    save_name = f"exp4_cycle_{cycle}.pt"
    
    # Consolidate from the CURRENT checkpoint (not baseline)
    consolidator = Consolidator(current_checkpoint, 'checkpoints/phase3')
    consolidator.sleep_cycle(replay_ratio_old=ratio_old, steps=150, save_name=save_name)
    
    current_checkpoint = os.path.join('checkpoints/phase3', save_name)
    
    # Evaluate
    eval_model = Consolidator(current_checkpoint, 'checkpoints/phase3').model
    L_old_after = evaluate_loss(eval_model, old_text, iters=100)
    L_new_after = evaluate_loss(eval_model, new_text, iters=100)
    
    results.append({
        'cycle': cycle,
        'L_old': L_old_after,
        'L_new': L_new_after
    })

print("\n" + "="*80)
print("Experiment 4: Repeated Sleep Cycles (75% New / 25% Old)")
print("="*80)
print(f"{'Cycle':<10} | {'L_old (Shakespeare)':<20} | {'L_new (Memory)':<20}")
print("-" * 80)
print(f"{'Baseline':<10} | {L_old_before:<20.4f} | {L_new_before:<20.4f}")
for r in results:
    print(f"{r['cycle']:<10} | {r['L_old']:<20.4f} | {r['L_new']:<20.4f}")
print("="*80)
