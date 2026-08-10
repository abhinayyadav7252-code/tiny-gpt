import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dataset import encode

def build_stage1(english_path, hindi_path, output_bin_path):
    print("Loading deduplicated English data...")
    with open(english_path, 'r', encoding='utf-8') as f:
        en_docs = f.read().split('\n<|endoftext|>\n')
        en_docs = [d.strip() for d in en_docs if d.strip()]
        
    print("Loading deduplicated Hindi data...")
    with open(hindi_path, 'r', encoding='utf-8') as f:
        hi_docs = f.read().split('\n<|endoftext|>\n')
        hi_docs = [d.strip() for d in hi_docs if d.strip()]
        
    print(f"Loaded {len(en_docs)} English docs and {len(hi_docs)} Hindi docs.")
    
    # Shuffle independently
    random.seed(42)
    random.shuffle(en_docs)
    random.shuffle(hi_docs)
    
    # Target ratio for Stage 1: roughly 73% English, 27% Hindi (approx 3:1)
    print("Interleaving documents (Ratio 3 En : 1 Hi)...")
    mixed_docs = []
    
    en_idx = 0
    hi_idx = 0
    
    while en_idx < len(en_docs) or hi_idx < len(hi_docs):
        # Add 3 English docs
        for _ in range(3):
            if en_idx < len(en_docs):
                mixed_docs.append(en_docs[en_idx])
                en_idx += 1
                
        # Add 1 Hindi doc
        if hi_idx < len(hi_docs):
            mixed_docs.append(hi_docs[hi_idx])
            hi_idx += 1
            
    print(f"Total mixed documents: {len(mixed_docs)}")
    
    # Tokenization
    print("Tokenizing mixed corpus (this may take a few minutes)...")
    all_tokens = []
    
    # We add a special token or newline to separate documents. 
    # For now, we just join with \n\n
    full_text = "\n\n".join(mixed_docs)
    
    # In a real large-scale system, we would stream this. Since this is a pilot, 
    # we can tokenize in chunks to show progress.
    chunk_size = 1000000
    
    for i in tqdm(range(0, len(full_text), chunk_size), desc="Tokenizing chunks"):
        chunk = full_text[i:i+chunk_size]
        tokens = encode(chunk)
        all_tokens.extend(tokens)
        
    print(f"\nTotal tokens in Stage 1 Corpus: {len(all_tokens):,}")
    
    # Save to binary
    print(f"Saving to {output_bin_path}...")
    os.makedirs(os.path.dirname(output_bin_path), exist_ok=True)
    
    tokens_np = np.array(all_tokens, dtype=np.uint16)
    
    # We use memmap to write efficiently
    m = np.memmap(output_bin_path, dtype=np.uint16, mode='w+', shape=(len(tokens_np),))
    m[:] = tokens_np[:]
    m.flush()
    
    print("Phase 10.2: Stage 1 Dataset (En+Hi) Build Complete!")
    print(f"You can now run pretraining on this dataset using: --data_path {output_bin_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_path", type=str, default="data/processed/stage1/english.txt")
    parser.add_argument("--hi_path", type=str, default="data/processed/stage1/hindi.txt")
    parser.add_argument("--out_bin", type=str, default="data/stage1_train.bin")
    args = parser.parse_args()
    
    build_stage1(args.en_path, args.hi_path, args.out_bin)
