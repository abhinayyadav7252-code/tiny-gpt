import argparse
import os
import glob
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC

def train_custom_tokenizer(data_dir: str, vocab_size: int, output_path: str):
    """
    Trains a custom BPE Tokenizer specifically on our mixed corpus 
    to handle Hindi + Hinglish + English without massive fragmentation.
    """
    print(f"Initializing BPE Tokenizer training. Target vocab size: {vocab_size}")
    
    # Initialize tokenizer with BPE model
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # Normalize unicode (very important for Devanagari/Hindi)
    tokenizer.normalizer = NFKC()
    
    # Pre-tokenize on whitespace
    tokenizer.pre_tokenizer = Whitespace()
    
    # Special tokens for our Cognitive Core (System, Tools, Padding, Unknown)
    special_tokens = [
        "[UNK]", "[PAD]", "[BOS]", "[EOS]", 
        "<|system|>", "<|user|>", "<|model|>", 
        "<|tool_call|>", "<|tool_result|>", 
        "<|memory_query|>", "<|memory_result|>"
    ]
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True
    )
    
    # Find all text data files to train on (e.g. from data/ folder)
    # We will assume user provides a .txt or .jsonl containing the corpus
    files = glob.glob(os.path.join(data_dir, "*.txt")) + glob.glob(os.path.join(data_dir, "*.jsonl"))
    
    if not files:
        print(f"No .txt or .jsonl files found in {data_dir}. Please provide a corpus.")
        return
        
    print(f"Training on files: {files}")
    tokenizer.train(files, trainer)
    
    # Save the tokenizer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)
    print(f"Tokenizer successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing corpus files (.txt/.jsonl)")
    parser.add_argument("--vocab_size", type=int, default=16384, help="Target vocabulary size")
    parser.add_argument("--output_path", type=str, default="checkpoints/chaitanya_bpe.json", help="Path to save tokenizer")
    
    args = parser.parse_args()
    train_custom_tokenizer(args.data_dir, args.vocab_size, args.output_path)
