import torch
import torch.nn as nn
from torch.nn import functional as F
import config
from dataset import vocab_size

class Head(nn.Module):
    """ One head of causal self-attention """
    def __init__(self, head_size):
        super().__init__()
        # Explicit Q, K, V implementation as requested
        self.key = nn.Linear(config.embed_dim, head_size, bias=False)
        self.query = nn.Linear(config.embed_dim, head_size, bias=False)
        self.value = nn.Linear(config.embed_dim, head_size, bias=False)
        
        # Causal mask (tril)
        self.register_buffer('tril', torch.tril(torch.ones(config.context_length, config.context_length)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, layer_past=None):
        B, T, C = x.shape
        
        k = self.key(x)   # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
            
        present = (k, v)
        
        # Attention scores: QK^T / sqrt(d)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5) # (B, T_q, T_k)
        
        # Causal mask: only apply if T_q > 1 (not generating single tokens)
        if T > 1:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Aggregate values
        out = wei @ v     # (B, T, head_size)
        return out, present

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, layer_past=None):
        if layer_past is None:
            layer_past = [None] * len(self.heads)
            
        outs = []
        presents = []
        for h, past in zip(self.heads, layer_past):
            out, present = h(x, past)
            outs.append(out)
            presents.append(present)
            
        out = torch.cat(outs, dim=-1)
        out = self.dropout(self.proj(out))
        return out, presents

class FeedForward(nn.Module):
    """ Simple linear layer followed by non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class MoELayer(nn.Module):
    """ Mixture of Experts Layer """
    def __init__(self, n_embd, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([FeedForward(n_embd) for _ in range(num_experts)])
        self.router = nn.Linear(n_embd, num_experts, bias=False)
        
    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C) # (B*T, C)
        
        router_logits = self.router(x_flat) # (B*T, num_experts)
        routing_weights = F.softmax(router_logits, dim=1) # (B*T, num_experts)
        
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=1)
        routing_weights = routing_weights / routing_weights.sum(dim=1, keepdim=True)
        
        final_output = torch.zeros_like(x_flat)
        
        # Load balancing auxiliary loss
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=0).sum(dim=0)
        router_prob_per_expert = F.softmax(router_logits, dim=1).mean(dim=0)
        
        fraction_of_tokens = tokens_per_expert / (B * T * self.top_k)
        aux_loss = self.num_experts * torch.sum(fraction_of_tokens * router_prob_per_expert)
        
        for i, expert in enumerate(self.experts):
            expert_idx, kth_expert = torch.where(selected_experts == i)
            if len(expert_idx) > 0:
                expert_inputs = x_flat[expert_idx]
                expert_outputs = expert(expert_inputs)
                weights = routing_weights[expert_idx, kth_expert].unsqueeze(1)
                final_output[expert_idx] += expert_outputs * weights
                
        return final_output.view(B, T, C), aux_loss

class Block(nn.Module):
    """ Transformer block: communication (attention) followed by computation (FFWD) """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        if getattr(config, 'use_moe', False):
            self.ffwd = MoELayer(n_embd, config.num_experts, config.top_k_experts)
        else:
            self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, layer_past=None):
        # Residual connections with LayerNorm
        sa_out, present = self.sa(self.ln1(x), layer_past)
        x = x + sa_out
        
        aux_loss = 0.0
        if getattr(config, 'use_moe', False):
            ffwd_out, aux_loss = self.ffwd(self.ln2(x))
        else:
            ffwd_out = self.ffwd(self.ln2(x))
            
        x = x + ffwd_out
        return x, present, aux_loss

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        # Token Embedding + Position Embedding
        self.token_embedding_table = nn.Embedding(vocab_size, config.embed_dim)
        self.position_embedding_table = nn.Embedding(config.context_length, config.embed_dim)
        
        self.blocks = nn.Sequential(*[Block(config.embed_dim, n_head=config.num_heads) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, vocab_size)

    def forward(self, idx, targets=None, past_key_values=None):
        B, T = idx.shape

        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)
            past_length = 0
        else:
            past_length = past_key_values[0][0][0].shape[-2]

        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        positions = torch.arange(past_length, past_length + T, device=config.device)
        pos_emb = self.position_embedding_table(positions) # (T, C)
        
        x = tok_emb + pos_emb # (B, T, C)
        
        presents = []
        total_aux_loss = 0.0
        for block, past in zip(self.blocks, past_key_values):
            x, present, aux_loss = block(x, past)
            presents.append(present)
            total_aux_loss += aux_loss
            
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            if getattr(config, 'use_moe', False):
                loss += config.moe_loss_coef * total_aux_loss

        return logits, loss, presents

    def generate(self, idx, max_new_tokens, use_cache=False):
        # Autoregressive generation
        from dataset import eos_token_id
        B = idx.shape[0]
        unfinished = torch.ones(B, dtype=torch.bool, device=idx.device)
        past_key_values = None
        for _ in range(max_new_tokens):
            if use_cache and past_key_values is not None:
                idx_cond = idx[:, -1:]
            else:
                idx_cond = idx[:, -config.context_length:]
                
            logits, loss, past_key_values = self(idx_cond, past_key_values=past_key_values if use_cache else None)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx_next = torch.where(unfinished.unsqueeze(1), idx_next, torch.tensor(eos_token_id, device=idx.device))
            unfinished = unfinished & (idx_next.squeeze(1) != eos_token_id)
            
            idx = torch.cat((idx, idx_next), dim=1)
            
            if not unfinished.any():
                break
        return idx

    def generate_with_confidence(self, idx, max_new_tokens, use_cache=False):
        # Autoregressive generation that returns the mean log probability (confidence)
        from dataset import eos_token_id
        B = idx.shape[0]
        log_probs = torch.zeros(B, device=idx.device)
        lengths = torch.zeros(B, device=idx.device)
        unfinished = torch.ones(B, dtype=torch.bool, device=idx.device)
        past_key_values = None
        for _ in range(max_new_tokens):
            if use_cache and past_key_values is not None:
                idx_cond = idx[:, -1:]
            else:
                idx_cond = idx[:, -config.context_length:]
                
            logits, loss, past_key_values = self(idx_cond, past_key_values=past_key_values if use_cache else None)
            logits = logits[:, -1, :] # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            
            # Sample the next token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx_next = torch.where(unfinished.unsqueeze(1), idx_next, torch.tensor(eos_token_id, device=idx.device))
            
            # Calculate log probability of the sampled token
            sampled_prob = probs.gather(1, idx_next)
            log_prob = torch.log(sampled_prob + 1e-9).squeeze(-1)
            log_probs += log_prob * unfinished.float()
            lengths += unfinished.float()
            
            unfinished = unfinished & (idx_next.squeeze(1) != eos_token_id)
            
            idx = torch.cat((idx, idx_next), dim=1)
            
            if not unfinished.any():
                break
            
        # Mean Log Probability of the generated sequence
        # Avoid division by zero by clamping lengths to at least 1
        mean_log_probs = (log_probs / lengths.clamp(min=1)).tolist()
        return idx, mean_log_probs

if __name__ == '__main__':
    from dataset import get_batch
    print("--- Test 3: Model Shape Verification ---")
    model = TinyGPT().to(config.device)
    xb, yb = get_batch()
    logits, loss, _ = model(xb, yb)
    
    print(f"Input X shape: {xb.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Expected shape: [{config.batch_size}, {config.context_length}, {vocab_size}] (But flattened by cross_entropy internal view if loss is computed)")
    
    # Let's get raw logits shape without loss calculation for clarity
    logits_raw, _, _ = model(xb)
    print(f"Raw Logits shape: {logits_raw.shape}")
    
    print(f"Initial Loss: {loss.item():.4f}")
    print("\n[OK] Model structure and forward pass verified.")
