from brain.episodic_memory import EpisodicMemory
from brain.semantic_memory import SemanticMemory
from brain.procedural_memory import ProceduralMemory
from brain.event_bus import bus, EventType
import logging

class MemoryManager:
    """
    Orchestrates the 3-Layer Hybrid Memory System and manages the Memory Lifecycle:
    Input -> Episodic -> Verification -> Semantic.
    """
    def __init__(self):
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()
        self.logger = logging.getLogger("ChaitanyaEventBus")
        
        # Subscribe to events
        bus.subscribe(EventType.USER_INPUT, self._handle_user_input)
        bus.subscribe(EventType.ANSWER_VERIFIED, self._handle_verification)
        
    def _handle_user_input(self, event_type, payload):
        """Record unverified claims into Episodic Memory."""
        user_text = payload.get("text", "")
        if user_text:
            # We record this as an episodic event
            record_id = self.episodic.record_experience(user_text, source="user")
            bus.publish(EventType.MEMORY_CREATED, {"type": "episodic", "id": record_id, "content": user_text})
            
    def _handle_verification(self, event_type, payload):
        """Promote verified facts to Semantic Memory."""
        fact = payload.get("fact")
        source = payload.get("source", "verified_internal")
        confidence = payload.get("confidence", 1.0)
        
        if fact and confidence > 0.8:
            # Move to semantic memory
            record_id = self.semantic.add_fact(fact, source)
            self.logger.info(f"Promoted to Semantic Memory: {fact}")
            bus.publish(EventType.MEMORY_PROMOTED, {"type": "semantic", "id": record_id, "content": fact})
            
    def search_all(self, query: str):
        """Retrieves relevant context from all memory layers."""
        ep_results = self.episodic.search(query, top_k=5)
        sem_results = self.semantic.search(query, top_k=3)
        proc_results = self.procedural.search_skills(query, top_k=1)
        
        return {
            "episodic": ep_results,
            "semantic": sem_results,
            "procedural": proc_results
        }
        
    def close(self):
        self.episodic.db.close()
        self.semantic.db.close()
        self.procedural.db.close()
