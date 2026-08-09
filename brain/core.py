import re
import torch
import config
import os
from model import TinyGPT
from dataset import encode, decode
from .memory import WorkingMemory, LongTermMemory
from .retrieval import retrieve
from .tools import safe_calc
from .self_model import SelfModel
from .metacognition import MetacognitiveEvaluator

class AIBrain:
    def __init__(self, use_self_model=True, use_confidence=True, use_verification=True):
        self.working_memory = WorkingMemory()
        self.long_term_memory = LongTermMemory()
        
        # Phase 4 components
        self.use_self_model = use_self_model
        self.use_confidence = use_confidence
        self.use_verification = use_verification
        
        self.self_model = SelfModel()
        self.metacognition = MetacognitiveEvaluator()
        
        # Load Phase 2B Generative Model
        self.model = TinyGPT()
        checkpoint_path = 'checkpoints/phase2_baseline.pt'
        if os.path.exists(checkpoint_path):
            try:
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
            except Exception as e:
                print(f"[Brain] Could not load checkpoint due to shape mismatch (expected during local testing Phase 5): {e}")
        self.model.eval()
        self.model.to(config.device)
        
    def process_query(self, user_query):
        if self.use_self_model:
            self.self_model.update_goal(user_query)
            
        self.working_memory.add("User", user_query)
        
        save_match = re.search(r'(?i)remember that my ([\w\s]+) is ([\w\s]+)', user_query)
        if save_match:
            key = save_match.group(1).strip()
            value = save_match.group(2).strip()
            self.long_term_memory.add_fact(key, value)
            if self.use_self_model:
                self.self_model.add_known(f"{key}: {value}")
            response = f"[Brain: Saved fact -> {key}: {value}] Got it, I will remember that."
            self.working_memory.add("AI", response)
            return response

        prompt_str = f"User: {user_query}\nAI:"
        
        # Decide if we need System 2 (Routing)
        needs_system2 = any(kw in user_query.lower() for kw in ["calculate", "solve", "math", "+", "-", "*", "/", "logic", "think", "who", "what", "where", "when", "why", "how", "capital", "invent"])
        
        if needs_system2:
            ai_response = self.system2_reasoning(prompt_str, user_query)
        else:
            ai_response = self.system1_generation(prompt_str, user_query)

        self.working_memory.add("AI", ai_response)
        return ai_response
        
    def system1_generation(self, prompt_str, user_query):
        context = torch.tensor(encode(prompt_str), dtype=torch.long, device=config.device).unsqueeze(0)
        
        if self.use_self_model:
            generated_idx, mean_log_probs = self.model.generate_with_confidence(context, max_new_tokens=40)
            mean_log_prob = mean_log_probs[0]
        else:
            generated_idx = self.model.generate(context, max_new_tokens=40)
            mean_log_prob = 0
            
        generated_idx = generated_idx[0].tolist()
        from dataset import eos_token_id
        prompt_len = len(encode(prompt_str))
        if eos_token_id in generated_idx[prompt_len:]:
            # we must only look for eos in the newly generated tokens
            try:
                eos_idx = generated_idx.index(eos_token_id, prompt_len)
                generated_idx = generated_idx[:eos_idx]
            except ValueError:
                pass
        
        # Simple Intent Router for Math Bypass
        math_keywords = ["add", "subtract", "multiply", "divide", "how many", "how much", "mph", "train", "hour", "baker", "dozen", "earns", "students", "buy", "sell"]
        if user_query and any(kw in user_query.lower() for kw in math_keywords) and any(c.isdigit() for c in user_query):
            needs_rag = False
            
        new_tokens = generated_idx[prompt_len:]
        ai_response = decode(new_tokens).strip()
        ai_response = ai_response.replace("<|endoftext|>", "").strip()
        
        # Tools
        calc_match = re.search(r'\[CALC:\s*([^\]]+)\]', ai_response)
        if calc_match:
            expression = calc_match.group(1).strip()
            result = safe_calc(expression)
            ai_response = ai_response.replace(f"[CALC: {expression}]", f"[CALC: {expression} = {result}]")
            if "The answer is" not in ai_response:
                ai_response += f" The answer is {result}."
                
        # RAG Tool
        from .tools import search_knowledge_base
        ret_match = re.search(r'\[RETRIEVE:\s*([^\]]+)\]', ai_response)
        if ret_match:
            ret_query = ret_match.group(1).strip()
            result = search_knowledge_base(ret_query)
            ai_response = ai_response.replace(f"[RETRIEVE: {ret_query}]", f"\n{result}\n")
            
        if not ai_response:
             ai_response = "I couldn't process that properly."
             
        # Metacognitive Evaluation
        if self.use_self_model:
            confidence, error_detected, error_reason = self.metacognition.evaluate_output(
                user_query, ai_response, mean_log_prob
            )
            
            if not self.use_confidence:
                confidence = 1.0
            if not self.use_verification:
                error_detected = False
                error_reason = None
            
            is_correct = not error_detected if any(k.lower() in user_query.lower() for k in self.metacognition.objective_truths.keys()) else None
            
            self.self_model.record_confidence(confidence, is_correct)
            
            if error_detected:
                self.self_model.record_error(error_reason)
                print(f"[Metacognition] System 1 failed verification: {error_reason}. Routing to System 2...")
                return self.system2_reasoning(prompt_str, user_query)
                
        return ai_response

    def system2_reasoning(self, prompt_str, user_query):
        print("[System 2] Activated. Generating multiple reasoning paths...")
        num_candidates = 3
        
        # Determine RAG Early (English + Hinglish keywords)
        factual_keywords = ["who", "what", "where", "when", "why", "how", "capital", "invent", "president", 
                            "kaun", "kya", "kahan", "kab", "kyon", "kaise", "rajdhani", "kisne"]
        needs_rag = any(kw in user_query.lower() for kw in factual_keywords)
        
        # Simple Intent Router for Math Bypass (English + Hinglish keywords)
        math_keywords = ["add", "subtract", "multiply", "divide", "how many", "how much", "mph", "train", "hour", "baker", "dozen", "earns", "students", "buy", "sell",
                         "jodo", "ghatao", "guna", "bhag", "kitne", "kitna", "bache", "paas"]
        is_math = user_query and any(kw in user_query.lower() for kw in math_keywords) and any(c.isdigit() for c in user_query)
        if is_math:
            needs_rag = True
            
        rag_context = ""
        from .tools import search_knowledge_base
        
        if needs_rag:
            if is_math:
                print("[System 2] Math detected. Bypassing external retrieval.")
                rag_context = "Retrieved Context: None."
            else:
                print("[System 2] External knowledge deemed necessary. Retrieving...")
                rag_context = search_knowledge_base(user_query)
                print(f"[System 2] {rag_context}")
            
            # Use exact format seen during SFT: User -> AI Retrieve -> System Context -> AI Answer
            search_query = user_query.split()[-1] if user_query else "unknown"
            prompt_str = f"User: {user_query}\nAI: [RETRIEVE] {search_query} [/RETRIEVE]\nSystem: {rag_context}\nAI:"
        else:
            prompt_str = f"User: {user_query}\nAI:"
            
        print(f"[DEBUG] System 2 prompt_str:\n{repr(prompt_str)}\n")
        context = torch.tensor(encode(prompt_str), dtype=torch.long, device=config.device).unsqueeze(0)
        context = context.repeat(num_candidates, 1)
        
        if self.use_self_model:
            generated_idx, mean_log_probs = self.model.generate_with_confidence(context, max_new_tokens=40)
        else:
            generated_idx = self.model.generate(context, max_new_tokens=40)
            mean_log_probs = [0] * num_candidates
            
        candidates = []
        from dataset import eos_token_id
        prompt_len = len(encode(prompt_str))
        
        for i in range(num_candidates):
            gen_idx = generated_idx[i].tolist()
            if eos_token_id in gen_idx[prompt_len:]:
                try:
                    eos_idx = gen_idx.index(eos_token_id, prompt_len)
                    gen_idx = gen_idx[:eos_idx]
                except ValueError:
                    pass
            
            # Decode ONLY the newly generated tokens
            new_tokens = gen_idx[prompt_len:]
            response = decode(new_tokens).strip()
            response = response.replace("<|endoftext|>", "").strip()
            candidates.append({'response': response, 'confidence': mean_log_probs[i]})
            
        print(f"[System 2] Generated {num_candidates} candidates.")
        
        best_candidate = None
        for idx, cand in enumerate(candidates):
            response = cand['response']
            
            # Math Tools
            calc_match = re.search(r'\[CALC:\s*([^\]]+)\]', response)
            if calc_match:
                expression = calc_match.group(1).strip()
                result = safe_calc(expression)
                response = response.replace(f"[CALC: {expression}]", f"[CALC: {expression} = {result}]")
                if "The answer is" not in response:
                    response += f" The answer is {result}."
                    
            # RAG Tools (Fallback if not caught early)
            ret_match = re.search(r'\[RETRIEVE:\s*([^\]]+)\]', response)
            if ret_match:
                ret_query = ret_match.group(1).strip()
                result = search_knowledge_base(ret_query)
                response = response.replace(f"[RETRIEVE: {ret_query}]", f"\n{result}\n")
                
            # Evidence Verification
            if needs_rag and rag_context:
                if "None." in rag_context:
                    # Unanswerable, any generated refusal is fine, skip verification
                    has_support = True
                else:
                    import string
                    ignore_words = {"based", "retrieved", "context", "from", "according", "think", "step", "have", "enough", "evidence"}
                    clean_response = response.lower().translate(str.maketrans('', '', string.punctuation))
                    keywords = [w for w in clean_response.split() if w not in ignore_words and len(w) > 2]
                    has_support = any(kw in rag_context.lower() for kw in keywords) if keywords else True
                
                if not has_support:
                    print(f"[System 2] Verification Failed: Candidate {idx+1} contradicts or lacks evidence from retrieved context.")
                    continue
                    
            if self.use_self_model:
                confidence, error_detected, error_reason = self.metacognition.evaluate_output(
                    user_query, response, cand['confidence']
                )
                
                if not error_detected:
                    print(f"[System 2] Verification passed on Candidate {idx+1}.")
                    best_candidate = response
                    break
            else:
                best_candidate = response
                break
                
        if best_candidate:
            return best_candidate
        else:
            print("[System 2] All paths failed verification. Falling back to highest confidence.")
            best_cand = max(candidates, key=lambda x: x['confidence'])
            return best_cand['response']
