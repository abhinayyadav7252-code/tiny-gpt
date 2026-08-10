import os
from brain.hybrid_db import HybridDB

class SemanticMemory:
    def __init__(self, db_dir="data"):
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "semantic.sqlite")
        self.db = HybridDB(db_path, "semantic_memory")
        
    def add_fact(self, fact: str, source: str = "verified_internal"):
        metadata = {
            "type": "fact",
            "source": source,
            "confidence": "high",
            "verification": "verified"
        }
        return self.db.add(fact, metadata)
        
    def search(self, query: str, top_k: int = 5):
        return self.db.search(query, top_k)
