import os
import random
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataset import encode

def load_docs(path):
    if not os.path.exists(path):
        print(f"Warning: File {path} not found.")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        docs = f.read().split('\n<|endoftext|>\n')
        return [d.strip() for d in docs if d.strip()]

def build_stage2(output_bin_path):
    print("Loading datasets for Stage 2 Curriculum Mixture...")
    
    # Stage 1 Replay
    en_docs = load_docs("data/raw/stage1/english.txt")
    hi_docs = load_docs("data/raw/stage1/hindi.txt")
    
    # Take a 30% sample of Stage 1 to prevent catastrophic forgetting
    # We'll take 50,000 En docs and 10,000 Hi docs for replay
    random.seed(42)
    random.shuffle(en_docs)
    random.shuffle(hi_docs)
    en_replay = en_docs[:50000]
    hi_replay = hi_docs[:10000]
    
    # Stage 2 New Data
    hinglish_docs = load_docs("data/raw/stage2/hinglish.txt")
    math_docs = load_docs("data/raw/stage2/math.txt")
    code_docs = load_docs("data/raw/stage2/code.txt")
    facts_docs = load_docs("data/raw/stage2/facts.txt")
    
    print(f"Loaded:")
    print(f"  - Stage 1 Replay (En): {len(en_replay)} docs")
    print(f"  - Stage 1 Replay (Hi): {len(hi_replay)} docs")
    print(f"  - Hinglish: {len(hinglish_docs)} docs")
    print(f"  - Math: {len(math_docs)} docs")
    print(f"  - Code: {len(code_docs)} docs")
    print(f"  - Facts: {len(facts_docs)} docs")
    
    # Mix all documents
    # A simple approach is to put them all in a list and shuffle
    # This gives an independent identically distributed (i.i.d) mixture
    mixed_docs = en_replay + hi_replay + hinglish_docs + math_docs + code_docs + facts_docs
    random.shuffle(mixed_docs)
    
    print(f"Total mixed documents for Stage 2: {len(mixed_docs)}")
    
    # Tokenization
    print("Tokenizing mixed corpus (this may take a few minutes)...")
    all_tokens = []
    
    # We join with \n\n to separate documents contextually
    full_text = "\n\n".join(mixed_docs)
    
    chunk_size = 1000000
    for i in tqdm(range(0, len(full_text), chunk_size), desc="Tokenizing chunks"):
        chunk = full_text[i:i+chunk_size]
        tokens = encode(chunk)
        all_tokens.extend(tokens)
        
    print(f"\nTotal tokens in Stage 2 Corpus: {len(all_tokens):,}")
    
    print(f"Saving to {output_bin_path}...")
    all_tokens_np = np.array(all_tokens, dtype=np.uint16)
    all_tokens_np.tofile(output_bin_path)
    print("Phase 11.0: Stage 2 Dataset Build Complete!")
    print(f"You can now run continual pretraining using: --train_data {output_bin_path} --resume checkpoints/50M/stage1_final.pt")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    build_stage2("data/stage2_train.bin")
