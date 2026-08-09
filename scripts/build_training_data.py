import json
import random

def build_mixture():
    print("Building clean training mixture...")
    
    # 1. Pretraining Data (General text)
    pretrain = [
        {"text": "The quick brown fox jumps over the lazy dog. This is a classic pangram."},
        {"text": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability."},
    ]
    
    # 2. Instruction Data
    instruction = [
        {"prompt": "User: What is the capital of France?\nAI:", "completion": " The capital of France is Paris."},
        {"prompt": "User: Explain gravity.\nAI:", "completion": " Gravity is a fundamental interaction which causes mutual attraction between all things with mass or energy."},
    ]
    
    # 3. Math Data
    math = [
        {"prompt": "User: What is 5 + 7?\nAI:", "completion": " 5 + 7 = 12.\nThe answer is 12."},
        {"prompt": "User: If John has 3 apples and buys 4 more, how many does he have?\nAI:", "completion": " John starts with 3 apples. He buys 4 more. 3 + 4 = 7.\nThe answer is 7."},
    ]
    
    # 4. RAG / Tool-Use Data (Crucial for System 2 behavior)
    rag = [
        {
            "prompt": "User: Who developed the theory of relativity?\nAI:", 
            "completion": " [RETRIEVE] theory of relativity [/RETRIEVE]\nSystem: Retrieved Context: Albert Einstein developed the theory of relativity.\nAI: Based on the retrieved context, Albert Einstein developed the theory of relativity."
        },
        {
            "prompt": "User: What is the largest planet in the solar system?\nAI:", 
            "completion": " [RETRIEVE] largest planet solar system [/RETRIEVE]\nSystem: Retrieved Context: Jupiter is the largest planet in the Solar System.\nAI: Jupiter is the largest planet in our solar system."
        }
    ]
    
    # Combine everything into a unified SFT format
    # For a causal LM, we typically train on "prompt + completion" and only calculate loss on the completion tokens.
    # Here we just create a combined text string for standard next-token prediction
    dataset = []
    
    for item in pretrain:
        dataset.append({"text": item["text"], "source": "pretrain"})
        
    for item in instruction + math + rag:
        dataset.append({"text": f"{item['prompt']}{item['completion']}", "source": "sft"})
        
    random.shuffle(dataset)
    
    output_path = "data/mixed_training_data.jsonl"
    with open(output_path, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Successfully built {len(dataset)} training examples at {output_path}")

if __name__ == "__main__":
    build_mixture()
