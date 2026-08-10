import json
from brain.event_bus import bus, EventType
from brain.memory_manager import MemoryManager
from brain.executive import ExecutiveController
from brain.self_model import SelfModel2
from brain.world_model import WorldModel
from brain.learning_loop import LearningLoop
from brain.consolidate import BackgroundConsolidator
from brain.regression_gate import RegressionGate
import logging

class ChaitanyaBrain:
    """
    The Cognitive Core initialized for Chaitanya v2.
    It wires together the EventBus and the cognitive subsystems.
    """
    def __init__(self, load_model=False, checkpoint_path='checkpoints/tiny_gpt_real.pt'):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ChaitanyaBrain")
        
        # Optionally load Neural Engine
        self.model = None
        self.encode = None
        self.decode = None
        self.device = 'cpu'
        
        if load_model:
            try:
                import torch
                import config
                from model import TinyGPT
                from dataset import encode, decode
                
                self.device = config.device
                self.model = TinyGPT()
                
                ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                if 'model_state' in ckpt:
                    self.model.load_state_dict(ckpt['model_state'])
                else:
                    self.model.load_state_dict(ckpt)
                    
                self.model.eval()
                self.model.to(self.device)
                self.encode = encode
                self.decode = decode
                self.logger.info(f"Loaded Neural Engine from {checkpoint_path}")
            except Exception as e:
                self.logger.warning(f"Could not load Neural Engine: {e}. Running in Mock Mode.")
        
        # 1. Initialize Subsystems
        self.memory_manager = MemoryManager()
        self.executive = ExecutiveController(
            self.memory_manager, 
            model=self.model, 
            encode_fn=self.encode, 
            decode_fn=self.decode, 
            device=self.device
        )
        self.self_model = SelfModel2()
        self.world_model = WorldModel()
        self.learning_loop = LearningLoop()
        self.consolidator = BackgroundConsolidator()
        self.regression_gate = RegressionGate()
        
        # Internal state to hold the answer before returning to user
        self._current_answer = None
        
        # Subscribe to final output events
        bus.subscribe(EventType.ANSWER_GENERATED, self._on_answer_generated)
        
        self.logger.info("Chaitanya Cognitive Core v2 initialized.")

    def _on_answer_generated(self, event_type, payload):
        self._current_answer = payload.get("answer")

    def process(self, user_input: str) -> str:
        """
        Entry point for the cognitive engine.
        Publishes the USER_INPUT event to the EventBus.
        """
        self._current_answer = None
        self.logger.info(f"Processing input: {user_input}")
        
        # Trigger the cognitive architecture
        bus.publish(EventType.USER_INPUT, {"text": user_input})
        
        # In a real async system, we would await a future here.
        # Since our mock ExecutiveController ticks synchronously, 
        # the answer will be populated.
        if self._current_answer:
            return self._current_answer
            
        return "I encountered a cognitive blockage and could not generate a response."

    def report_error(self, error_msg: str, context: dict = None):
        """Allows external systems (or user) to report an error for learning."""
        bus.publish(EventType.ERROR_DETECTED, {"error": error_msg, "context": context or {}})
        return "Error reported to the Learning Loop."

    def shutdown(self):
        """Cleanly shutdown memory connections."""
        self.memory_manager.close()

if __name__ == "__main__":
    # Test harness
    brain = ChaitanyaBrain()
    print(brain.process("What is the capital of India?"))
    brain.shutdown()
