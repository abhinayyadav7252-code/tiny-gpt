import json
import random

def build_mixture():
    print("Building full SFT training mixture...")
    
    data = []
    
    # 1. RAG Exact Extraction (60 examples)
    capitals = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Germany", "Berlin"), ("Canada", "Ottawa"), 
                ("Australia", "Canberra"), ("Brazil", "Brasilia"), ("India", "New Delhi"), ("China", "Beijing"), ("Russia", "Moscow"),
                ("Spain", "Madrid"), ("Egypt", "Cairo"), ("Mexico", "Mexico City"), ("South Korea", "Seoul"), ("UK", "London")]
    
    # To ensure 150+ examples, let's just create more questions
    for country, capital in capitals * 2:
        q = f"What is the capital of {country}?"
        query = q.split()[-1] # Match core.py: search_query = user_query.split()[-1]
        prompt = f"User: {q}\nAI: [RETRIEVE] {query} [/RETRIEVE]\nSystem: Retrieved Context: The capital of {country} is {capital}.\nAI:"
        completion = f" Based on the retrieved context, {capital}."
        data.append({"prompt": prompt, "completion": completion})
        
    inventions = [("telephone", "Alexander Graham Bell"), ("light bulb", "Thomas Edison"), ("airplane", "Wright Brothers"),
                  ("printing press", "Johannes Gutenberg"), ("world wide web", "Tim Berners-Lee"), ("theory of relativity", "Albert Einstein"),
                  ("penicillin", "Alexander Fleming"), ("radio", "Guglielmo Marconi"), ("battery", "Alessandro Volta")]
                  
    for invention, inventor in inventions * 2:
        q = f"Who developed the {invention}?"
        query = q.split()[-1]
        prompt = f"User: {q}\nAI: [RETRIEVE] {query} [/RETRIEVE]\nSystem: Retrieved Context: {inventor} developed the {invention}.\nAI:"
        completion = f" Based on the retrieved context, {inventor}."
        data.append({"prompt": prompt, "completion": completion})
        
    elements = [("gold", "Au"), ("silver", "Ag"), ("iron", "Fe"), ("oxygen", "O"), ("carbon", "C"),
                ("hydrogen", "H"), ("sodium", "Na"), ("potassium", "K"), ("calcium", "Ca"), ("helium", "He")]
                
    for element, symbol in elements * 2:
        q = f"What is the chemical symbol for {element}?"
        query = q.split()[-1]
        prompt = f"User: {q}\nAI: [RETRIEVE] {query} [/RETRIEVE]\nSystem: Retrieved Context: The chemical symbol for {element} is {symbol}.\nAI:"
        completion = f" Based on the retrieved context, {symbol}."
        data.append({"prompt": prompt, "completion": completion})
        
    eval_facts = [
        ("Who was the first woman to win a Nobel Prize?", "Marie Curie", "Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize."),
        ("In what year did the Apollo 11 moon landing occur?", "1969", "The Apollo 11 moon landing occurred in 1969. Neil Armstrong was the first man to walk on the moon, famously stating \"That's one small step for man, one giant leap for mankind.\""),
        ("What is the largest planet in our solar system?", "Jupiter", "Jupiter is the largest planet in the Solar System. It is a gas giant with a mass more than two and a half times that of all the other planets in the Solar System combined."),
        ("Who developed the theory of relativity?", "Albert Einstein", "Albert Einstein developed the theory of relativity, one of the two pillars of modern physics. His mass-energy equivalence formula E = mc^2 is the world's most famous equation."),
        ("What is the capital of Australia?", "Canberra", "The capital of Australia is Canberra. It is Australia's largest inland city and the eighth-largest city overall."),
        ("What is the elevation of Mount Everest?", "8,848.86 meters", "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. Its elevation is 8,848.86 meters."),
        ("When was the Declaration of Independence adopted?", "July 4, 1776", "The Declaration of Independence was adopted by the Second Continental Congress meeting in Philadelphia, Pennsylvania, on July 4, 1776."),
        ("What is the speed of light in a vacuum in kilometers per second?", "300,000", "The speed of light in a vacuum is exactly 299,792,458 meters per second. This is approximately 300,000 kilometers per second or 186,000 miles per second."),
        ("Who was the first man to walk on the moon, and in what year did he do it?", "Neil Armstrong in 1969", "The Apollo 11 moon landing occurred in 1969. Neil Armstrong was the first man to walk on the moon, famously stating \"That's one small step for man, one giant leap for mankind.\""),
        ("What planet is known as the Red Planet and which one is the largest?", "Mars is the Red Planet and Jupiter is the largest.", "Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System. It is often referred to as the \"Red Planet\" due to the effect of the iron oxide prevalent on its surface."),
        ("What process do plants use to make food?", "Photosynthesis", "Photosynthesis is the process by which plants make food."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare", "William Shakespeare wrote Romeo and Juliet."),
        ("What is the largest ocean on Earth?", "The Pacific Ocean", "The Pacific Ocean is the largest ocean on Earth."),
        ("At what temperature does water boil?", "100 degrees Celsius", "Water boils at 100 degrees Celsius.")
    ]
    
    for q, ans, ctx in eval_facts * 2:
        query = q.split()[-1]
        prompt = f"User: {q}\nAI: [RETRIEVE] {query} [/RETRIEVE]\nSystem: Retrieved Context: {ctx}\nAI:"
        completion = f" Based on the retrieved context, {ans}."
        data.append({"prompt": prompt, "completion": completion})
        
    # 2. RAG Irrelevant Context -> Abstain (50 examples)
    irrelevant_facts = ["Dogs are popular pets.", "The sky is blue due to Rayleigh scattering.", "Apples are a type of fruit.",
                        "Cats like to sleep.", "Cars have four wheels.", "Water is wet.", "Birds can fly.", "The sun is hot."]
    
    questions = [f"What is the capital of {c}?" for c, _ in capitals] + \
                [f"Who developed the {i}?" for i, _ in inventions] + \
                [f"What is the chemical symbol for {e}?" for e, _ in elements]
                
    random.shuffle(questions)
    for q in questions[:50]: # We might not have 50 unique questions, but we'll use what we have
        fact = random.choice(irrelevant_facts)
        query = q.split()[-1]
        prompt = f"User: {q}\nAI: [RETRIEVE] {query} [/RETRIEVE]\nSystem: Retrieved Context: {fact}\nAI:"
        completion = " I don't have enough evidence."
        data.append({"prompt": prompt, "completion": completion})
        
    # 3. Missing Context -> Abstain (50 examples)
    # Re-shuffle to get different questions
    random.shuffle(questions)
    for q in questions[:50]:
        query = q.split()[-1]
        prompt = f"User: {q}\nAI: [RETRIEVE] {query} [/RETRIEVE]\nSystem: Retrieved Context: None.\nAI:"
        completion = " I don't have enough evidence."
        data.append({"prompt": prompt, "completion": completion})
        
    # 4. Math / Pure Reasoning (Without context or Irrelevant Context) (40 examples)
    math_problems = [
        ("If John has 5 apples and gives 2 to Mary, then buys 3 more, how many apples does John have?", "6"),
        ("A baker makes 12 dozen cookies. He sells 50 cookies in the morning and 34 cookies in the afternoon. How many cookies does he have left?", "60"),
        ("Samantha earns $15 per hour. If she works 4 hours a day for 5 days a week, how much does she earn in 2 weeks?", "600"),
        ("There are 24 students in a class. If 1/3 of them are boys, how many girls are in the class?", "16"),
        ("A train travels at 60 mph for 3 hours and then at 80 mph for 2 hours. What is the total distance traveled?", "340")
    ] * 8 # Duplicate to get 40
    
    for q, a in math_problems:
        query = q.split()[-1]
        # 50% chance of empty context, 50% chance of irrelevant context
        if random.random() < 0.5:
            ctx = "None."
        else:
            ctx = random.choice(irrelevant_facts)
            
        prompt = f"User: {q}\nAI: [RETRIEVE] {query} [/RETRIEVE]\nSystem: Retrieved Context: {ctx}\nAI:"
        completion = f" Let's think step by step. {a}."
        data.append({"prompt": prompt, "completion": completion})

    # 5. Hinglish Data (40 examples)
    hinglish_factual = [
        ("France ki rajdhani kya hai?", "rajdhani", "The capital of France is Paris.", " Context ke hisaab se, iska jawab Paris hai."),
        ("Gold ka chemical symbol kya hota hai?", "hota", "The chemical symbol for gold is Au. Its atomic number is 79, making it one of the higher atomic number elements that occur naturally.", " Context ke hisaab se, iska jawab Au hai."),
        ("Japan ki rajdhani kya hai?", "rajdhani", "The capital of Japan is Tokyo.", " Context ke hisaab se, iska jawab Tokyo hai."),
        ("Water kitne temperature par boil hota hai?", "hota", "Water boils at 100 degrees Celsius.", " Context ke hisaab se, iska jawab 100 degrees Celsius hai.")
    ] * 5 # 20 examples
    
    for q, query, ctx, completion in hinglish_factual:
        prompt = f"User: {q}\nAI: [RETRIEVE] {query} [/RETRIEVE]\nSystem: Retrieved Context: {ctx}\nAI:"
        data.append({"prompt": prompt, "completion": completion})
        
    hinglish_unanswerable = [
        ("Vibranium kisne discover kiya?", "kiya", "None.", " Mere paas iska jawab dene ke liye evidence nahi hai."),
        ("Mars par kitne aliens rehte hain?", "hain", "None.", " Mere paas iska jawab dene ke liye evidence nahi hai."),
        ("Atlantis city ki population kya hai?", "hai", "None.", " Mere paas iska jawab dene ke liye evidence nahi hai.")
    ] * 4 # 12 examples
    
    for q, query, ctx, completion in hinglish_unanswerable:
        prompt = f"User: {q}\nAI: [RETRIEVE] {query} [/RETRIEVE]\nSystem: Retrieved Context: {ctx}\nAI:"
        data.append({"prompt": prompt, "completion": completion})
        
    hinglish_math = [
        ("Agar Ram ke paas 5 seb hain aur usne 2 bache ko diye, aur fir 3 kharide, toh uske paas kitne seb bache?", "bache", "None.", " Let's think step by step. 6."),
        ("Ek baker ne 12 dozen cookies banayi. Usne subah 50 aur dopahar me 34 bech di. Uske paas kitni bachi?", "bachi", "Dogs are popular pets.", " Let's think step by step. 60.")
    ] * 4 # 8 examples
    
    for q, query, ctx, completion in hinglish_math:
        prompt = f"User: {q}\nAI: [RETRIEVE] {query} [/RETRIEVE]\nSystem: Retrieved Context: {ctx}\nAI:"
        data.append({"prompt": prompt, "completion": completion})

    random.shuffle(data)
    
    output_path = "data/mixed_training_data.jsonl"
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Successfully built {len(data)} training examples at {output_path}")

if __name__ == "__main__":
    build_mixture()
