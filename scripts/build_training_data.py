import json
import random

def build_mixture():
    print("Building full SFT training mixture...")
    
    data = []
    
    # 1. RAG Exact Extraction (60 examples)
    capitals = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Germany", "Berlin"), ("Canada", "Ottawa"), 
                ("Australia", "Canberra"), ("Brazil", "Brasilia"), ("India", "New Delhi"), ("China", "Beijing"), ("Russia", "Moscow"),
                ("Spain", "Madrid"), ("Egypt", "Cairo"), ("Mexico", "Mexico City"), ("South Korea", "Seoul"), ("UK", "London")]
    
    for country, capital in capitals:
        prompt = f"User: What is the capital of {country}?\nAI: [RETRIEVE] {country}? [/RETRIEVE]\nSystem: Retrieved Context: The capital of {country} is {capital}.\nAI:"
        completion = f" Based on the retrieved context, {capital}."
        data.append({"prompt": prompt, "completion": completion})
        
    inventions = [("telephone", "Alexander Graham Bell"), ("light bulb", "Thomas Edison"), ("airplane", "Wright Brothers"),
                  ("printing press", "Johannes Gutenberg"), ("world wide web", "Tim Berners-Lee"), ("theory of relativity", "Albert Einstein"),
                  ("penicillin", "Alexander Fleming"), ("radio", "Guglielmo Marconi"), ("battery", "Alessandro Volta")]
                  
    for invention, inventor in inventions:
        prompt = f"User: Who developed the {invention}?\nAI: [RETRIEVE] {invention}? [/RETRIEVE]\nSystem: Retrieved Context: {inventor} developed the {invention}.\nAI:"
        completion = f" Based on the retrieved context, {inventor}."
        data.append({"prompt": prompt, "completion": completion})

    elements = [("gold", "Au"), ("silver", "Ag"), ("iron", "Fe"), ("oxygen", "O"), ("carbon", "C"),
                ("hydrogen", "H"), ("sodium", "Na"), ("potassium", "K"), ("calcium", "Ca"), ("helium", "He")]
                
    for element, symbol in elements:
        prompt = f"User: What is the chemical symbol for {element}?\nAI: [RETRIEVE] {element}? [/RETRIEVE]\nSystem: Retrieved Context: The chemical symbol for {element} is {symbol}.\nAI:"
        completion = f" Based on the retrieved context, {symbol}."
        data.append({"prompt": prompt, "completion": completion})
        
    # 2. RAG Irrelevant Context -> Abstain (40 examples)
    # We mix random questions with random irrelevant facts.
    irrelevant_facts = ["Dogs are popular pets.", "The sky is blue due to Rayleigh scattering.", "Apples are a type of fruit.",
                        "Cats like to sleep.", "Cars have four wheels.", "Water is wet.", "Birds can fly.", "The sun is hot."]
    
    questions = [f"What is the capital of {c}?" for c, _ in capitals] + \
                [f"Who developed the {i}?" for i, _ in inventions] + \
                [f"What is the chemical symbol for {e}?" for e, _ in elements]
                
    random.shuffle(questions)
    for q in questions[:40]:
        fact = random.choice(irrelevant_facts)
        query = q.split()[-1].strip("?")
        prompt = f"User: {q}\nAI: [RETRIEVE] {query}? [/RETRIEVE]\nSystem: Retrieved Context: {fact}\nAI:"
        completion = " I don't have enough evidence."
        data.append({"prompt": prompt, "completion": completion})
        
    # 3. Missing Context -> Abstain (40 examples)
    for q in questions[40:80]:
        query = q.split()[-1].strip("?")
        prompt = f"User: {q}\nAI: [RETRIEVE] {query}? [/RETRIEVE]\nSystem: Retrieved Context: None.\nAI:"
        completion = " I don't have enough evidence."
        data.append({"prompt": prompt, "completion": completion})
        
    # 4. Math / Pure Reasoning (Without context) (20 examples)
    math_problems = [
        ("If John has 5 apples and gives 2 to Mary, then buys 3 more, how many apples does John have?", "6 apples"),
        ("A baker makes 12 dozen cookies. He sells 50 cookies in the morning and 34 cookies in the afternoon. How many cookies does he have left?", "60 cookies"),
        ("Samantha earns $15 per hour. If she works 4 hours a day for 5 days a week, how much does she earn in 2 weeks?", "$600"),
        ("There are 24 students in a class. If 1/3 of them are boys, how many girls are in the class?", "16 girls"),
        ("A train travels at 60 mph for 3 hours and then at 80 mph for 2 hours. What is the total distance traveled?", "340 miles")
    ] * 4 # Duplicate to get 20
    
    for q, a in math_problems:
        query = q.split()[-1].strip("?")
        prompt = f"User: {q}\nAI: [RETRIEVE] {query}? [/RETRIEVE]\nSystem: Retrieved Context: None.\nAI:"
        completion = f" Let's think step by step. {a}."
        data.append({"prompt": prompt, "completion": completion})

    random.shuffle(data)
    
    output_path = "data/mixed_training_data.jsonl"
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Successfully built {len(data)} training examples at {output_path}")

if __name__ == "__main__":
    build_mixture()
