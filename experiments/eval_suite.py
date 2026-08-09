import json
import time
import os
import sys

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.core import AIBrain

def load_dataset(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_evaluation(brain, dataset, exp_name):
    print(f"\n{'='*50}\nRunning {exp_name}\n{'='*50}")
    
    total = len(dataset)
    correct = 0
    total_latency = 0
    
    # We will measure confidence calibration
    # If correct, what was confidence?
    # If incorrect, what was confidence?
    conf_correct = []
    conf_incorrect = []

    for item in dataset:
        query = item.get('question') or item.get('query')
        expected = item.get('expected_answer') or item.get('answer')
        
        start_time = time.time()
        
        # Override printing to avoid spam
        # In a real eval suite we might silence stdout
        response = brain.process_query(query)
        
        latency = time.time() - start_time
        total_latency += latency
        
        # Simple exact match/substring match
        # Extract the expected final answer for GSM8K (after ####)
        if "####" in expected:
            expected = expected.split("####")[1].strip()
            
        is_correct = expected.lower() in response.lower()
        
        if is_correct:
            correct += 1
            
        print(f"Q: {query}")
        print(f"A: {response}")
        print(f"Expected: {expected} | Correct: {is_correct}\n")
        
    accuracy = correct / total * 100
    avg_latency = total_latency / total
    
    print(f"--- Results for {exp_name} ---")
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print(f"Avg Latency: {avg_latency:.2f}s")
    return accuracy

def main():
    print("Loading datasets...")
    gsm8k_data = load_dataset('data/eval_gsm8k_subset.json')
    rag_data = load_dataset('data/eval_rag_subset.json')
    
    results = {}
    
    # Base Model (System 1, no verification)
    brain_base = AIBrain(use_self_model=False, use_confidence=False, use_verification=False)
    # Monkeypatch to disable System 2
    brain_base.process_query = lambda q: brain_base.system1_generation(f"User: {q}\nAI:", q)
    results['Exp A: Base (Math)'] = run_evaluation(brain_base, gsm8k_data, "Exp A: Base Model (Math)")
    
    # System 2 Only
    brain_sys2 = AIBrain(use_self_model=True, use_confidence=True, use_verification=True)
    # Disable RAG (Tools already used inside System 2)
    # We will test Sys2 on Math
    results['Exp B: Sys2 Only (Math)'] = run_evaluation(brain_sys2, gsm8k_data, "Exp B: System 2 Only (Math)")
    
    # RAG Only (System 1 + RAG)
    brain_rag = AIBrain(use_self_model=False, use_confidence=False, use_verification=False)
    # We test on RAG dataset, Sys 1 handles it via `[RETRIEVE]` regex
    results['Exp C: RAG Only (Factual)'] = run_evaluation(brain_rag, rag_data, "Exp C: RAG Only (Factual)")
    
    # Full Brain
    brain_full = AIBrain(use_self_model=True, use_confidence=True, use_verification=True)
    results['Exp E: Full Brain (Factual)'] = run_evaluation(brain_full, rag_data, "Exp E: Full Brain (Factual)")
    
    print("\n================ SUMMARY ================")
    for k, v in results.items():
        print(f"{k}: {v:.1f}%")

if __name__ == "__main__":
    main()
