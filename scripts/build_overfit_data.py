import json

def generate_overfit_dataset():
    data = []
    
    # 1. Relevant Context -> Answer (10 examples)
    facts = [
        ("Apollo 11 landed on the Moon in 1969.", "When did Apollo 11 land on the Moon?", "1969."),
        ("The capital of France is Paris.", "What is the capital of France?", "Paris."),
        ("Water boils at 100 degrees Celsius.", "At what temperature does water boil?", "100 degrees Celsius."),
        ("Albert Einstein developed the theory of relativity.", "Who developed the theory of relativity?", "Albert Einstein."),
        ("The chemical symbol for gold is Au.", "What is the chemical symbol for gold?", "Au."),
        ("Jupiter is the largest planet in the Solar System.", "Which planet is the largest in the Solar System?", "Jupiter."),
        ("Mount Everest is the highest mountain on Earth.", "What is the highest mountain on Earth?", "Mount Everest."),
        ("William Shakespeare wrote Romeo and Juliet.", "Who wrote Romeo and Juliet?", "William Shakespeare."),
        ("The Pacific Ocean is the largest ocean on Earth.", "What is the largest ocean on Earth?", "The Pacific Ocean."),
        ("Photosynthesis is the process by which plants make food.", "What process do plants use to make food?", "Photosynthesis.")
    ]
    
    for context, question, answer in facts:
        prompt = f"User: {question}\nAI: [RETRIEVE] {question.split()[-1]} [/RETRIEVE]\nSystem: Retrieved Context: {context}\nAI:"
        completion = f" Based on the retrieved context, {answer}"
        data.append({"prompt": prompt, "completion": completion})
        
    # 2. Irrelevant Context -> Ignore/General Answer if known, or Abstain (5 examples)
    irrelevant = [
        ("Dogs are popular pets.", "When did Apollo 11 land on the Moon?", "I don't have enough evidence"),
        ("The sky is blue due to Rayleigh scattering.", "What is the capital of France?", "I don't have enough evidence"),
        ("Apples are a type of fruit.", "At what temperature does water boil?", "I don't have enough evidence"),
        ("Cats like to sleep.", "Who developed the theory of relativity?", "I don't have enough evidence"),
        ("Cars have four wheels.", "What is the chemical symbol for gold?", "I don't have enough evidence"),
    ]
    
    for context, question, answer in irrelevant:
        prompt = f"User: {question}\nAI: [RETRIEVE] {question.split()[-1]} [/RETRIEVE]\nSystem: Retrieved Context: {context}\nAI:"
        completion = f" {answer}."
        data.append({"prompt": prompt, "completion": completion})

    # 3. Missing Evidence -> Abstain (5 examples)
    missing = [
        ("Who discovered the fictional element Vibranium?", "I don't have enough evidence"),
        ("What is the population of the underwater city of Atlantis?", "I don't have enough evidence"),
        ("How many aliens live on Mars?", "I don't have enough evidence"),
        ("Who is the current king of the Sun?", "I don't have enough evidence"),
        ("What is the exact number of grains of sand on Earth?", "I don't have enough evidence"),
    ]
    
    for question, answer in missing:
        prompt = f"User: {question}\nAI: [RETRIEVE] {question.split()[-1]} [/RETRIEVE]\nSystem: Retrieved Context: None.\nAI:"
        completion = f" {answer}."
        data.append({"prompt": prompt, "completion": completion})

    output_path = "data/sft_overfit_data.jsonl"
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Created {len(data)} overfit sanity examples at {output_path}")

if __name__ == "__main__":
    generate_overfit_dataset()
