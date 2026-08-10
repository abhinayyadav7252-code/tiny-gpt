import os
from brain.hybrid_db import HybridDB

class ProceduralMemory:
    def __init__(self, db_dir="data"):
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "procedural.sqlite")
        self.db = HybridDB(db_path, "procedural_memory")
        
    def add_skill(self, trigger_condition: str, action_sequence: str):
        content = f"IF {trigger_condition} THEN {action_sequence}"
        metadata = {
            "type": "skill",
            "trigger": trigger_condition
        }
        return self.db.add(content, metadata)
        
    def search_skills(self, context_query: str, top_k: int = 2):
        return self.db.search(context_query, top_k)
