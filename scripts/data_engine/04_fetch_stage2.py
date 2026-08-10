import os
import json
from datasets import load_dataset
from tqdm import tqdm

def fetch_hinglish(output_path, num_samples=100000):
    print("Fetching Hinglish Data (L3Cube-Pune Hinglish Tweets)...")
    dataset = load_dataset("l3cube-pune/hinglish-tweet", split="train")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        count = 0
        for item in tqdm(dataset, total=min(num_samples, len(dataset))):
            text = item.get("tweet", "").strip()
            if text:
                f.write(text + "\n<|endoftext|>\n")
                count += 1
            if count >= num_samples:
                break
    print(f"Saved {count} Hinglish documents to {output_path}")

def fetch_math(output_path, num_samples=8000):
    print("Fetching Math Data (GSM8K)...")
    dataset = load_dataset("gsm8k", "main", split="train")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        count = 0
        for item in tqdm(dataset, total=min(num_samples, len(dataset))):
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            if question and answer:
                # Format to natural language structure
                formatted_text = f"Question: {question}\nStep-by-step solution: {answer}"
                f.write(formatted_text + "\n<|endoftext|>\n")
                count += 1
            if count >= num_samples:
                break
    print(f"Saved {count} Math documents to {output_path}")

def fetch_code(output_path, num_samples=1000):
    print("Fetching Code Data (MBPP)...")
    dataset = load_dataset("mbpp", split="train")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        count = 0
        for item in tqdm(dataset, total=min(num_samples, len(dataset))):
            prompt = item.get("text", "").strip()
            code = item.get("code", "").strip()
            if prompt and code:
                # Format to natural language explanation
                formatted_text = f"Python Task: {prompt}\nCode:\n```python\n{code}\n```"
                f.write(formatted_text + "\n<|endoftext|>\n")
                count += 1
            if count >= num_samples:
                break
    print(f"Saved {count} Code documents to {output_path}")

def fetch_facts(output_path, num_samples=50000):
    print("Fetching Factual Data (Wikitext-2)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        count = 0
        for item in tqdm(dataset, total=min(num_samples, len(dataset))):
            text = item.get("text", "").strip()
            # Ignore empty lines or very short lines (like headers)
            if len(text) > 50:
                f.write(text + "\n<|endoftext|>\n")
                count += 1
            if count >= num_samples:
                break
    print(f"Saved {count} Factual documents to {output_path}")

if __name__ == "__main__":
    os.makedirs("data/raw/stage2", exist_ok=True)
    
    print("\n" + "="*50)
    print("Phase 11.0: Stage 2 Curriculum Fetching")
    print("="*50)
    
    fetch_hinglish("data/raw/stage2/hinglish.txt")
    fetch_math("data/raw/stage2/math.txt")
    fetch_code("data/raw/stage2/code.txt")
    fetch_facts("data/raw/stage2/facts.txt")
    
    print("\nStage 2 Curriculum Fetching Complete!")
