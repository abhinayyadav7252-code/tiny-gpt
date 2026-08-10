import os
import argparse
import hashlib
from tqdm import tqdm
import re

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    print("Please install datasketch: pip install datasketch")
    exit(1)

def get_exact_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    # Simple word tokenization for MinHash
    tokens = set(re.findall(r'\w+', text.lower()))
    for d in tokens:
        m.update(d.encode('utf8'))
    return m

def process_file(input_path, output_path, lsh, exact_hashes, threshold=0.8):
    print(f"\nProcessing {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    docs = content.split('\n<|endoftext|>\n')
    docs = [d.strip() for d in docs if d.strip()]
    
    kept_docs = []
    exact_dupes = 0
    near_dupes = 0
    
    for i, doc in enumerate(tqdm(docs, desc=f"Deduplicating")):
        # 1. Exact Match Deduplication
        doc_hash = get_exact_hash(doc)
        if doc_hash in exact_hashes:
            exact_dupes += 1
            continue
            
        exact_hashes.add(doc_hash)
        
        # 2. MinHash Near-Duplicate Detection
        m = get_minhash(doc)
        
        # Check if near duplicate exists
        result = lsh.query(m)
        if result:
            near_dupes += 1
            continue
            
        # If unique, add to LSH and keep
        # Use filename + index as unique ID
        doc_id = f"{os.path.basename(input_path)}_{i}"
        lsh.insert(doc_id, m)
        kept_docs.append(doc)
        
    # Write deduped docs
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in kept_docs:
            f.write(doc + "\n<|endoftext|>\n")
            
    print(f"Stats for {os.path.basename(input_path)}:")
    print(f"  Original docs: {len(docs)}")
    print(f"  Exact duplicates removed: {exact_dupes}")
    print(f"  Near duplicates removed: {near_dupes}")
    print(f"  Final docs kept: {len(kept_docs)}")
    
    return exact_hashes, lsh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/raw/stage1")
    parser.add_argument("--output_dir", type=str, default="data/processed/stage1")
    parser.add_argument("--threshold", type=float, default=0.85, help="Jaccard similarity threshold for MinHash")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Global state across all languages/files to prevent cross-lingual duplicates where possible
    exact_hashes = set()
    lsh = MinHashLSH(threshold=args.threshold, num_perm=128)
    
    files_to_process = ["english.txt", "hindi.txt"]
    
    for filename in files_to_process:
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)
        
        if os.path.exists(input_path):
            exact_hashes, lsh = process_file(input_path, output_path, lsh, exact_hashes, args.threshold)
        else:
            print(f"File not found: {input_path}")
            
    print("\nPhase 10.1: Level 1 & 2 Deduplication (Exact + MinHash) Complete.")
    print("Note: Semantic Deduplication (Level 3) should be run separately on a GPU instance for performance.")

if __name__ == "__main__":
    main()
