import tiktoken

enc = tiktoken.get_encoding("gpt2")

sentences = {
    "English": "The computer is thinking.",
    "Hinglish": "AI ko ye sawaal kaise solve karna hai?",
    "Hindi": "यह सवाल कैसे हल करना है?"
}

with open('tokenizer_audit.txt', 'w', encoding='utf-8') as f:
    f.write("=== Tokenizer Audit (GPT-2 BPE) ===\n\n")
    for lang, text in sentences.items():
        tokens = enc.encode(text, allowed_special="all")
        # Ensure we decode individual tokens carefully since some might be broken utf-8 bytes
        decoded = [enc.decode([t], errors="replace") for t in tokens]
        
        char_len = len(text)
        word_len = len(text.split())
        token_len = len(tokens)
        
        f.write(f"[{lang}]\n")
        f.write(f"Text: {text}\n")
        f.write(f"Tokens ({token_len}): {decoded}\n")
        f.write(f"Tokens/Char: {token_len / char_len:.2f}\n")
        f.write(f"Tokens/Word: {token_len / word_len:.2f}\n")
        f.write("-" * 40 + "\n")

