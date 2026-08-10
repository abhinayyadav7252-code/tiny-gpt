import sqlite3
import numpy as np
import faiss
import json
import os
from sentence_transformers import SentenceTransformer

class HybridDB:
    """
    Combines SQLite (source of truth) with FAISS (fast semantic index).
    """
    def __init__(self, db_path: str, table_name: str, embedding_dim: int = 384):
        self.db_path = db_path
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        
        # Initialize SQLite
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_sqlite()
        
        # Initialize FAISS
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        # We need an ID mapping from FAISS index to SQLite ID
        self.id_map = faiss.IndexIDMap(self.index)
        
        # Initialize Embedding model
        # Using a small fast model for embeddings
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        self._load_faiss_from_sqlite()

    def _init_sqlite(self):
        cursor = self.conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                embedding BLOB
            )
        """)
        self.conn.commit()

    def _load_faiss_from_sqlite(self):
        """Loads all embeddings from SQLite into FAISS on startup."""
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT id, embedding FROM {self.table_name} WHERE embedding IS NOT NULL")
        rows = cursor.fetchall()
        
        if rows:
            ids = np.array([row['id'] for row in rows], dtype=np.int64)
            embeddings = np.array([np.frombuffer(row['embedding'], dtype=np.float32) for row in rows])
            self.id_map.add_with_ids(embeddings, ids)

    def add(self, content: str, metadata: dict) -> int:
        """Adds a record to SQLite and FAISS."""
        # Generate embedding
        emb = self.embedder.encode([content])[0].astype(np.float32)
        emb_bytes = emb.tobytes()
        
        cursor = self.conn.cursor()
        cursor.execute(f"""
            INSERT INTO {self.table_name} (content, metadata, embedding)
            VALUES (?, ?, ?)
        """, (content, json.dumps(metadata), emb_bytes))
        self.conn.commit()
        
        record_id = cursor.lastrowid
        
        # Add to FAISS
        self.id_map.add_with_ids(np.array([emb]), np.array([record_id], dtype=np.int64))
        return record_id

    def update_metadata(self, record_id: int, new_metadata: dict):
        """Updates metadata of a record in SQLite."""
        cursor = self.conn.cursor()
        cursor.execute(f"""
            UPDATE {self.table_name} SET metadata = ? WHERE id = ?
        """, (json.dumps(new_metadata), record_id))
        self.conn.commit()

    def delete(self, record_id: int):
        """Deletes a record from SQLite and FAISS."""
        cursor = self.conn.cursor()
        cursor.execute(f"DELETE FROM {self.table_name} WHERE id = ?", (record_id,))
        self.conn.commit()
        
        # Remove from FAISS
        self.id_map.remove_ids(np.array([record_id], dtype=np.int64))

    def search(self, query: str, top_k: int = 5):
        """Semantic search using FAISS, retrieves full rows from SQLite."""
        if self.id_map.ntotal == 0:
            return []
            
        emb = self.embedder.encode([query]).astype(np.float32)
        distances, indices = self.id_map.search(emb, top_k)
        
        results = []
        cursor = self.conn.cursor()
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1: continue # FAISS returns -1 for empty slots
            
            cursor.execute(f"SELECT id, content, metadata FROM {self.table_name} WHERE id = ?", (int(idx),))
            row = cursor.fetchone()
            if row:
                metadata = json.loads(row['metadata'])
                status = metadata.get('status', 'candidate')
                
                # Safeguard: Do not retrieve contradicted or deprecated memories
                if status in ['contradicted', 'deprecated']:
                    continue
                    
                results.append({
                    "id": row['id'],
                    "content": row['content'],
                    "metadata": metadata,
                    "distance": float(dist)
                })
        return results

    def get_all(self):
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT id, content, metadata FROM {self.table_name}")
        rows = cursor.fetchall()
        return [{"id": r["id"], "content": r["content"], "metadata": json.loads(r["metadata"])} for r in rows]

    def close(self):
        self.conn.close()
