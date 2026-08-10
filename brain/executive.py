from brain.working_memory import WorkingMemory
from brain.event_bus import bus, EventType
import logging

class ExecutiveController:
    """
    The main decision maker. Instead of hardcoded giant if/else blocks, it assesses 
    the state of the WorkingMemory and decides the next cognitive action based on a policy (currently rules-based).
    """
    def __init__(self, memory_manager, model=None, encode_fn=None, decode_fn=None, device='cpu'):
        self.wm = WorkingMemory()
        self.memory_manager = memory_manager
        self.logger = logging.getLogger("ChaitanyaEventBus")
        
        self.model = model
        self.encode = encode_fn
        self.decode = decode_fn
        self.device = device
        
        bus.subscribe(EventType.USER_INPUT, self.handle_user_input)
        
    def handle_user_input(self, event_type, payload):
        """Entry point for a new thought process."""
        user_query = payload.get("text", "")
        if not user_query: return
        
        # 1. State Assessment
        self.wm.clear()
        self.wm.update_goal(f"Answer user query: {user_query}")
        
        # Attempt to retrieve relevant memory Context
        memories = self.memory_manager.search_all(user_query)
        for ep in memories["episodic"]:
            self.wm.add_evidence(f"Episodic: {ep['content']}")
        for sem in memories["semantic"]:
            self.wm.add_fact(f"Semantic: {sem['content']}")
        for proc in memories["procedural"]:
            self.wm.add_constraint(f"Skill: {proc['content']}")
            
        bus.publish(EventType.MEMORY_RETRIEVED, {"memories": memories})
        
        # 2. Decide Action (Rule-based Policy for v2)
        self.tick()
        
    def tick(self):
        """Evaluates working memory and executes next step in the loop."""
        # Simple rule-based execution policy for now:
        # If we have a goal, generate a plan
        if not self.wm.state.plan:
            self._create_plan()
            return
            
        # If we have a plan but no tool results yet, we might need a tool
        if not self.wm.state.tool_results and any("math" in h.lower() for h in self.wm.state.hypotheses):
            self._call_tool("calculator", {"query": "evaluate"})
            return
            
        # Finally, generate an answer based on working memory
        self._generate_final_answer()
        
    def _create_plan(self):
        # In a real model, this would be a forward pass asking the model for a plan.
        # Here we mock the rule-based policy.
        self.wm.set_plan(["Analyze query", "Check memory", "Formulate response"])
        bus.publish(EventType.PLAN_CREATED, {"plan": self.wm.state.plan})
        self.tick()
        
    def _call_tool(self, tool_name, args):
        bus.publish(EventType.TOOL_CALLED, {"tool_name": tool_name, "args": args})
        # Tool execution happens asynchronously or by a tool manager.
        # For now, we mock the result.
        result = "Tool execution result mocked"
        self.wm.add_tool_result(tool_name, result)
        bus.publish(EventType.TOOL_RESULT, {"tool_name": tool_name, "result": result})
        self.tick()
        
    def _generate_final_answer(self):
        prompt = self.wm.state.to_prompt_context()
        
        if self.model and self.encode and self.decode:
            import torch
            try:
                # We limit context length to avoid breaking 1.5M model context window
                context_idx = self.encode(prompt)
                if len(context_idx) > 200:
                    context_idx = context_idx[-200:]
                
                context_tensor = torch.tensor(context_idx, dtype=torch.long, device=self.device).unsqueeze(0)
                out_idx = self.model.generate(context_tensor, max_new_tokens=50)[0].tolist()
                
                # Extract only the generated part
                final_answer = self.decode(out_idx[len(context_idx):]).strip()
                if not final_answer: final_answer = "[Empty Model Output]"
            except Exception as e:
                self.logger.error(f"Model generation failed: {e}")
                final_answer = "Generation Failed."
        else:
            final_answer = "This is a drafted response based on Working Memory. (Model Mocked)"
            
        self.wm.set_confidence(0.85)
        bus.publish(EventType.ANSWER_GENERATED, {"answer": final_answer, "confidence": self.wm.state.confidence})
        
        # Trigger verification
        self._verify_answer(final_answer)
        
    def _verify_answer(self, answer):
        # Verification layer
        is_verified = True # Mocking a verification pass
        if is_verified:
            bus.publish(EventType.ANSWER_VERIFIED, {"fact": answer, "source": "self-verified", "confidence": self.wm.state.confidence})
        else:
            bus.publish(EventType.ERROR_DETECTED, {"error": "Verification failed", "context": self.wm.state.to_dict()})
