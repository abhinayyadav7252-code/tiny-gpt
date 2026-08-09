import torch
import config
import os
import tiktoken

# Initialize BPE tokenizer (GPT-2 standard)
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
eos_token_id = 50256

def encode(s):
    return enc.encode(s, allowed_special="all")

def decode(l):
    return enc.decode(l)

# Default data path for Phase 5
data_path = os.path.join(os.path.dirname(__file__), 'data', 'general_text.txt')

def get_batch(split='train'):
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    data_split = train_data if split == 'train' else val_data
    if len(data_split) <= config.context_length:
        data_split = data_split.repeat((config.context_length // len(data_split)) + 2)
        
    ix = torch.randint(len(data_split) - config.context_length, (config.batch_size,))
    x = torch.stack([data_split[i:i+config.context_length] for i in ix])
    y = torch.stack([data_split[i+1:i+config.context_length+1] for i in ix])
    return x.to(config.device), y.to(config.device)
