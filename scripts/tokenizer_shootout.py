import argparse
import os
import tiktoken
from tokenizers import Tokenizer

def run_shootout(custom_tokenizer_path: str):
    """
    Compares fragmentation and compression across:
    1. Custom BPE (Trained on Hindi/Hinglish/English)
    2. GPT-2 Tokenizer (Standard English BPE)
    3. cl100k_base (GPT-4 / modern standard)
    """
    print("========================================")
    print("=== TOKENIZER SHOOTOUT ===")
    print("========================================\n")
    
    # 1. Load Tokenizers
    try:
        if not os.path.exists(custom_tokenizer_path):
            print(f"Error: Custom tokenizer not found at {custom_tokenizer_path}")
            print("Please run `python scripts/train_tokenizer.py` first.")
            return
        custom_tok = Tokenizer.from_file(custom_tokenizer_path)
        print("[OK] Loaded Custom Tokenizer")
    except Exception as e:
        print(f"Failed to load custom tokenizer: {e}")
        return
        
    gpt2_tok = tiktoken.get_encoding("gpt2")
    gpt4_tok = tiktoken.get_encoding("cl100k_base")
    print("[OK] Loaded GPT-2 and GPT-4 Tokenizers\n")
    
    # 2. Test Cases
    test_cases = {
        "English (Standard)": "The quick brown fox jumps over the lazy dog.",
        "Hinglish (Mixed)": "Bhai, ye neural network training kitna time legi? GPU melt na ho jaye.",
        "Hindi (Devanagari)": "चैतन्य एक कृत्रिम बुद्धिमत्ता है, जो इंसानी दिमाग की तरह सोचता है।",
        "Code/Tools (Syntax)": "<|tool_call|> {\"name\": \"calculator\", \"args\": {\"query\": \"15 * 24\"}} <|tool_result|>"
    }
    
    for label, text in test_cases.items():
        words = len(text.split())
        chars = len(text)
        
        # Tokenize
        custom_enc = custom_tok.encode(text).ids
        gpt2_enc = gpt2_tok.encode(text)
        gpt4_enc = gpt4_tok.encode(text)
        
        print(f"--- Test: {label} ---")
        print(f"Input: '{text}' (Words: {words}, Chars: {chars})")
        
        def print_stats(name, tokens):
            tokens_per_word = len(tokens) / max(1, words)
            print(f"  {name:15s} | Tokens: {len(tokens):3d} | Tokens/Word: {tokens_per_word:.2f}")
            
        print_stats("Custom BPE", custom_enc)
        print_stats("GPT-2 BPE", gpt2_enc)
        print_stats("GPT-4 BPE", gpt4_enc)
        print("")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, default="checkpoints/chaitanya_bpe.json")
    args = parser.parse_args()
    
    run_shootout(args.tokenizer_path)
