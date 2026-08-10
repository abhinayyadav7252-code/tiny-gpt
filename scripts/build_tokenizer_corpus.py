import os
from datasets import load_dataset
import random

def build_tokenizer_corpus(output_path="data/tokenizer_corpus.txt"):
    """
    Builds a large representative corpus for training the Custom BPE Tokenizer.
    Includes:
    - English (TinyStories subset)
    - Hindi (Wikipedia subset)
    - Hinglish (Synthetic/Mixed)
    - Code / Tool Syntax
    """
    print("Building large representative corpus for Tokenizer Training...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        
        # 1. English (TinyStories subset - 20,000 stories)
        print("Downloading English subset (TinyStories)...")
        try:
            ts = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            for i, example in enumerate(ts):
                if i >= 20000: break
                f.write(example["text"].replace("\n", " ") + "\n")
        except Exception as e:
            print(f"Warning: Failed to load TinyStories: {e}")
            
        # 2. Hindi (Wikipedia subset - 5,000 articles)
        print("Downloading Hindi subset (Wikipedia)...")
        try:
            hi_wiki = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train", streaming=True)
            for i, example in enumerate(hi_wiki):
                if i >= 5000: break
                # Write first 500 chars of each article to keep it balanced
                f.write(example["text"][:500].replace("\n", " ") + "\n")
        except Exception as e:
            print(f"Warning: Failed to load Hindi Wiki: {e}")
            
        # 3. Hinglish & Tool Syntax (Synthetic generation)
        print("Generating Hinglish & Tool Syntax...")
        hinglish_templates = [
            "Bhai, ye model train hone mein kitna time lega?",
            "Kal maine ek mast neural network code likha tha, GPU pe run kar raha hu.",
            "Chaitanya ek AI assistant hai jo Hindi aur English dono samajhta hai.",
            "Working Memory update karna zaroori hai context ke liye.",
            "Mujhe samajh nahi aa raha ki ye error kyu de raha hai, check kar na.",
            "Training loss kam ho gaya hai, ab validation dekhte hain."
        ]
        
        tool_templates = [
            '<|tool_call|> {"name": "calculator", "args": {"query": "15 * 24"}} <|tool_result|>',
            '<|tool_call|> {"name": "search", "args": {"query": "quantum computing"}} <|tool_result|>',
            '<|memory_query|> "user preferences" <|memory_result|>',
            '<|system|> You are Chaitanya, a cognitive AI. <|user|> Hello! <|model|>'
        ]
        
        # Amplify these synthetic patterns so the tokenizer learns them well
        for _ in range(5000):
            f.write(random.choice(hinglish_templates) + "\n")
            f.write(random.choice(tool_templates) + "\n")
            
    print(f"Successfully built representative corpus at {output_path}")

if __name__ == "__main__":
    build_tokenizer_corpus()
