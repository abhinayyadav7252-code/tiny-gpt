import json
import time
import os
import sys
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain.core import AIBrain
from brain.rag_system import get_retriever

def load_dataset(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_math_answer(text):
    """Adversarially hard math extraction to prevent substring false positives."""
    if text is None:
        return None
    
    # Try to find "The answer is X"
    match = re.search(r'(?i)the answer is\s*([-+]?\d*\.?\d+)', text)
    if match:
        return match.group(1).strip()
        
    # Fallback to the last number in the text
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if numbers:
        return numbers[-1]
    
    return None

def evaluate_math(brain, dataset, exp_name):
    print(f"\nEvaluating {exp_name} (Math)...")
    total = len(dataset)
    correct = 0
    
    for item in dataset:
        query = item.get('question') or item.get('query')
        expected = item.get('expected_answer') or item.get('answer')
        
        if "####" in expected:
            expected = expected.split("####")[1].strip()
            
        response = brain.process_query(query)
        extracted = extract_math_answer(response)
        
        # Exact string float matching to avoid '16' in '160'
        is_correct = False
        if extracted is not None:
            try:
                if float(extracted) == float(expected):
                    is_correct = True
            except ValueError:
                pass
                
        if is_correct:
            correct += 1
            
    acc = correct / total * 100
    return acc

def evaluate_retrieval_isolated(dataset, top_k_list=[1, 3, 5]):
    print("\nEvaluating RAG Retrieval (Isolated)...")
    retriever = get_retriever('hybrid')
    
    results = {k: 0 for k in top_k_list}
    total = len(dataset)
    
    for item in dataset:
        query = item['query']
        expected_context_keywords = item.get('expected_keywords', [])
        if not expected_context_keywords:
            # Fallback to checking if answer is in retrieved doc
            expected_context_keywords = [item['answer']]
            
        retrieved_docs = retriever.search(query, top_k=max(top_k_list))
        retrieved_texts = [d[0].lower() for d in retrieved_docs]
        
        for k in top_k_list:
            top_k_texts = retrieved_texts[:k]
            # Check if any expected keyword is in any of the top_k docs
            found = False
            for text in top_k_texts:
                if any(kw.lower() in text for kw in expected_context_keywords):
                    found = True
                    break
            if found:
                results[k] += 1
                
    recalls = {f"Recall@{k}": (results[k] / total) * 100 for k in top_k_list}
    return recalls

def evaluate_factual(brain, dataset, exp_name):
    print(f"\nEvaluating {exp_name} (Factual)...")
    total = len(dataset)
    correct = 0
    hallucinated = 0
    abstained = 0
    
    for item in dataset:
        query = item['query']
        expected = item['answer'].lower()
        is_unanswerable = item.get('unanswerable', False)
        
        response = brain.process_query(query).lower()
        
        if is_unanswerable:
            # Correct behavior is abstention/refusal
            if "don't know" in response or "not found" in response or "cannot" in response or "no relevant" in response:
                abstained += 1
                correct += 1
            else:
                hallucinated += 1
        else:
            if expected in response:
                correct += 1
            else:
                hallucinated += 1
                
    acc = correct / total * 100
    hallucination_rate = hallucinated / total * 100
    abstention_rate = abstained / sum(1 for item in dataset if item.get('unanswerable', False)) * 100 if any(item.get('unanswerable', False) for item in dataset) else 0.0
    
    return acc, hallucination_rate, abstention_rate

def main():
    print("Loading datasets...")
    gsm8k_data = load_dataset('data/eval_gsm8k_subset.json')
    rag_data = load_dataset('data/eval_rag_subset.json')
    
    print("\n--- PHASE 6.1: EVALUATION SUITE RUNNING ---")
    
    # 1. Isolated Retrieval Test
    retrieval_metrics = evaluate_retrieval_isolated(rag_data)
    
    # 2. Base Model (System 1)
    brain_base = AIBrain(use_self_model=False, use_confidence=False, use_verification=False)
    brain_base.process_query = lambda q: brain_base.system1_generation(f"User: {q}\nAI:", q)
    base_math_acc = evaluate_math(brain_base, gsm8k_data, "Base Model")
    base_fact_acc, base_halluc, _ = evaluate_factual(brain_base, rag_data, "Base Model")
    
    # 3. System 2 Only
    brain_sys2 = AIBrain(use_self_model=True, use_confidence=True, use_verification=True)
    sys2_math_acc = evaluate_math(brain_sys2, gsm8k_data, "System 2 Only")
    sys2_fact_acc, sys2_halluc, _ = evaluate_factual(brain_sys2, rag_data, "System 2 Only")
    
    # 4. Full Brain (Sys 2 + RAG + Tools)
    brain_full = AIBrain(use_self_model=True, use_confidence=True, use_verification=True)
    full_fact_acc, full_halluc, full_abstention = evaluate_factual(brain_full, rag_data, "Full Brain")
    
    print("\n================ ABLATION SUMMARY ================")
    print(f"{'Experiment':<20} | {'Math Acc':<10} | {'Fact Acc':<10} | {'Halluc %':<10} | {'Abstention %':<12}")
    print("-" * 75)
    print(f"{'Base Model':<20} | {base_math_acc:>8.1f}% | {base_fact_acc:>8.1f}% | {base_halluc:>8.1f}% | {'N/A':>12}")
    print(f"{'System 2 Only':<20} | {sys2_math_acc:>8.1f}% | {sys2_fact_acc:>8.1f}% | {sys2_halluc:>8.1f}% | {'N/A':>12}")
    print(f"{'Full Brain (RAG)':<20} | {'N/A':>10} | {full_fact_acc:>8.1f}% | {full_halluc:>8.1f}% | {full_abstention:>11.1f}%")
    
    print("\n================ RAG RETRIEVAL (ISOLATED) ================")
    for k, v in retrieval_metrics.items():
        print(f"{k}: {v:.1f}%")

if __name__ == "__main__":
    main()
