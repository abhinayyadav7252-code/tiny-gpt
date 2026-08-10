import os
import glob
import numpy as np
from tqdm import tqdm
import json
import sys

# Ensure we can import from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import encode, vocab_size

def prepare_data(corpus_path: str, output_dir: str = "data"):
    """
    Reads a large corpus (txt or jsonl), encodes it using the frozen Custom Tokenizer,
    and writes train.bin and val.bin as uint16 arrays for ultra-fast pretraining.
    """
    print(f"Preparing pretraining data from {corpus_path}")
    print(f"Active Tokenizer Vocab Size: {vocab_size}")
    
    if vocab_size > 65535:
        print("Error: Vocab size > 65535, cannot use uint16 for memmap. Need uint32.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    files = glob.glob(corpus_path) if '*' in corpus_path else [corpus_path]
    if not files:
        print(f"No files found at {corpus_path}")
        return
        
    all_tokens = []
    
    for file_path in files:
        print(f"Encoding {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.txt'):
                for line in tqdm(f):
                    if line.strip():
                        all_tokens.extend(encode(line.strip() + " [EOS]"))
            else: # json or jsonl
                for line in tqdm(f):
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        # Could be SFT data or Raw text
                        text = data.get("text") or data.get("prompt", "") + data.get("completion", "")
                        if text:
                            all_tokens.extend(encode(text.strip() + " [EOS]"))
                    except:
                        pass
                        
    # Split 90% train, 10% val
    n = int(0.9 * len(all_tokens))
    train_data = np.array(all_tokens[:n], dtype=np.uint16)
    val_data = np.array(all_tokens[n:], dtype=np.uint16)
    
    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")
    
    train_data.tofile(train_path)
    val_data.tofile(val_path)
    
    print(f"Saved {len(train_data):,} tokens to {train_path}")
    print(f"Saved {len(val_data):,} tokens to {val_path}")
    print("Ready for 25M Pretraining!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, default="data/tokenizer_corpus.txt", help="Path or glob to text files")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for .bin files")
    args = parser.parse_args()
    
    prepare_data(args.corpus_path, args.output_dir)
