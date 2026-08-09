from brain.core import AIBrain

def main():
    print("=====================================")
    print("🧠 Project Chaitanya: AI BRAIN (Phase 2A)")
    print("Type 'exit' or 'quit' to close.")
    print("=====================================\n")
    
    brain = AIBrain()
    
    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Shutting down AI Brain...")
                break
                
            response = brain.process_query(user_input)
            print(f"AI: {response}")
            
        except KeyboardInterrupt:
            print("\nShutting down AI Brain...")
            break
            
if __name__ == "__main__":
    main()
