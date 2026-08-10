import os
import argparse
from datasets import load_dataset
from tqdm import tqdm

def fetch_tinystories(output_file, max_docs=100000):
    print("Fetching TinyStories (High-quality English Prose)...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for item in tqdm(dataset, total=max_docs):
            text = item['text'].strip()
            if text:
                # Basic cleaning: remove extra whitespace but preserve paragraph structure
                text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
                f.write(text + "\n<|endoftext|>\n")
                count += 1
            if count >= max_docs:
                break
    print(f"Saved {count} English documents to {output_file}")

def fetch_hindi_wikipedia(output_file, max_docs=50000):
    print("Fetching Hindi Wikipedia (Native Devanagari)...")
    # Wikipedia 2023 is high quality native Hindi
    dataset = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train", streaming=True)
    
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for item in tqdm(dataset, total=max_docs):
            text = item['text'].strip()
            if text:
                # Remove common wikipedia boilerplate (this is basic, semantic filtering will catch the rest)
                lines = text.split("\n")
                clean_lines = [l.strip() for l in lines if l.strip() and not l.startswith("श्रेणी:")]
                clean_text = "\n".join(clean_lines)
                
                if len(clean_text) > 100: # Ensure it has some substance
                    f.write(clean_text + "\n<|endoftext|>\n")
                    count += 1
            if count >= max_docs:
                break
    print(f"Saved {count} Hindi documents to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_docs", type=int, default=200000, help="Number of English documents to fetch")
    parser.add_argument("--hi_docs", type=int, default=70000, help="Number of Hindi documents to fetch")
    parser.add_argument("--out_dir", type=str, default="data/raw/stage1", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    fetch_tinystories(os.path.join(args.out_dir, "english.txt"), max_docs=args.en_docs)
    fetch_hindi_wikipedia(os.path.join(args.out_dir, "hindi.txt"), max_docs=args.hi_docs)
    
    print("\nPhase 10.1: Raw Corpus Construction for Stage 1 Complete.")
