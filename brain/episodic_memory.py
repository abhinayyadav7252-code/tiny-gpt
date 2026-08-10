import os
from brain.hybrid_db import HybridDB

class EpisodicMemory:
    def __init__(self, db_dir="data"):
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "episodic.sqlite")
        self.db = HybridDB(db_path, "episodic_memory")
        
    def record_experience(self, content: str, source: str, confidence: str = "unknown"):
        metadata = {
            "type": "experience",
            "source": source,
            "confidence": confidence,
            "verification": "pending"
        }
        return self.db.add(content, metadata)
        
    def get_unverified_claims(self):
        return [r for r in self.db.get_all() if r["metadata"].get("verification") == "pending"]
        
    def mark_verified(self, record_id: int):
        # We fetch the row, and usually it will be promoted and deleted from here, or just updated
        pass
        
    def search(self, query: str, top_k: int = 3):
        return self.db.search(query, top_k)
