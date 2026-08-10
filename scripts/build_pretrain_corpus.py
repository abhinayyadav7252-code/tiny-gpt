import argparse
import os
import tiktoken
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Stream and tokenize pretraining corpus")
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--max_tokens", type=int, default=20_000_000, help="Target token count for training split")
    parser.add_argument("--validation_tokens", type=int, default=2_000_000, help="Target token count for validation split")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize BPE tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # We use uint16 because max gpt2 token is 50256
    dtype = np.uint16
    
    print(f"Loading {args.dataset} with streaming=True...")
    dataset = load_dataset(args.dataset, streaming=True, split="train")
    
    # Keep track of tokens
    train_tokens = 0
    val_tokens = 0
    
    train_file = os.path.join(args.output_dir, "train.bin")
    val_file = os.path.join(args.output_dir, "val.bin")
    
    # Open binary files for writing
    with open(train_file, 'wb') as f_train, open(val_file, 'wb') as f_val:
        pbar = tqdm(total=args.max_tokens + args.validation_tokens, desc="Processing Tokens")
        
        # Simple iterator
        iterator = iter(dataset)
        
        while train_tokens < args.max_tokens or val_tokens < args.validation_tokens:
            try:
                example = next(iterator)
            except StopIteration:
                print("Warning: Reached end of dataset before token budget met!")
                break
                
            text = example['text']
            
            # Encode text, append EOT token (50256)
            tokens = enc.encode(text, allowed_special="all")
            tokens.append(enc.eot_token)
            
            token_arr = np.array(tokens, dtype=dtype)
            
            # Decide where to put it
            if val_tokens < args.validation_tokens:
                f_val.write(token_arr.tobytes())
                val_tokens += len(tokens)
                pbar.update(len(tokens))
            elif train_tokens < args.max_tokens:
                f_train.write(token_arr.tobytes())
                train_tokens += len(tokens)
                pbar.update(len(tokens))
                
        pbar.close()
        
    print("\nDataset building complete!")
    print(f"Saved to: {args.output_dir}")
    print(f"Train Tokens: {train_tokens:,}")
    print(f"Val Tokens: {val_tokens:,}")

if __name__ == "__main__":
    main()
