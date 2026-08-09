import os
import json
import numpy as np

# Try importing FAISS and SentenceTransformers, but allow graceful fallback for TF-IDF only if unavailable
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

import math
from collections import Counter

class KnowledgeBase:
    """A generic knowledge base interface."""
    def __init__(self, documents):
        self.documents = documents
        
    def search(self, query, top_k=2):
        raise NotImplementedError("Search not implemented.")

class TFIDFRetriever(KnowledgeBase):
    """Lexical baseline retriever using TF-IDF (Dependency-Free)."""
    def __init__(self, documents):
        super().__init__(documents)
        self.doc_words = [doc.lower().split() for doc in documents]
        self.vocab = set(w for doc in self.doc_words for w in doc)
        
        # DF
        self.df = Counter()
        for doc in self.doc_words:
            for w in set(doc):
                self.df[w] += 1
                
        self.N = len(documents)
        
    def _tf_idf(self, words):
        vec = {}
        tf = Counter(words)
        for w, count in tf.items():
            if w in self.df:
                idf = math.log((1 + self.N) / (1 + self.df[w])) + 1
                vec[w] = count * idf
        return vec
        
    def search(self, query, top_k=2):
        q_words = query.lower().split()
        q_vec = self._tf_idf(q_words)
        
        q_norm = math.sqrt(sum(v*v for v in q_vec.values()))
        if q_norm == 0:
            return []
            
        results = []
        for idx, doc_words in enumerate(self.doc_words):
            d_vec = self._tf_idf(doc_words)
            d_norm = math.sqrt(sum(v*v for v in d_vec.values()))
            if d_norm == 0:
                continue
                
            dot = sum(q_vec.get(w, 0) * d_vec.get(w, 0) for w in self.vocab)
            sim = dot / (q_norm * d_norm)
            if sim > 0:
                results.append((self.documents[idx], sim))
                
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

class FAISSRetriever(KnowledgeBase):
    """Semantic retriever using FAISS and small embeddings."""
    def __init__(self, documents, model_name='all-MiniLM-L6-v2'):
        super().__init__(documents)
        if not FAISS_AVAILABLE:
            raise ImportError("faiss and sentence_transformers are required for FAISS Retriever.")
            
        self.model = SentenceTransformer(model_name)
        
        # Encode documents
        print("Encoding knowledge base...")
        doc_embeddings = self.model.encode(self.documents, show_progress_bar=False)
        self.doc_embeddings = np.array(doc_embeddings).astype('float32')
        
        # Build FAISS Index
        dim = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.doc_embeddings)
        
    def search(self, query, top_k=2):
        query_embedding = self.model.encode([query]).astype('float32')
        
        # Search index
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            if idx != -1:
                # We invert distance to a similarity score (lower distance = higher sim)
                # For L2, just storing distance as score (lower is better) but returning it as negative for unified ranking later
                score = -distances[0][i] 
                results.append((self.documents[idx], score))
        return results

class HybridRetriever(KnowledgeBase):
    """Combines TF-IDF and FAISS retrieval."""
    def __init__(self, documents):
        super().__init__(documents)
        self.tfidf = TFIDFRetriever(documents)
        if FAISS_AVAILABLE:
            self.faiss = FAISSRetriever(documents)
        else:
            self.faiss = None
            
    def search(self, query, top_k=2):
        if not self.faiss:
            return self.tfidf.search(query, top_k)
            
        tfidf_results = self.tfidf.search(query, top_k=5)
        faiss_results = self.faiss.search(query, top_k=5)
        
        # Simple Reciprocal Rank Fusion or Score combination
        # For simplicity, we just aggregate and pick top_k unique docs
        seen = set()
        combined = []
        
        # We interleave the results
        for i in range(5):
            if i < len(faiss_results):
                doc = faiss_results[i][0]
                if doc not in seen:
                    seen.add(doc)
                    combined.append((doc, "FAISS"))
            if i < len(tfidf_results):
                doc = tfidf_results[i][0]
                if doc not in seen:
                    seen.add(doc)
                    combined.append((doc, "TFIDF"))
                    
            if len(combined) >= top_k:
                break
                
        return combined[:top_k]

def load_corpus(knowledge_dir='knowledge'):
    docs = []
    if not os.path.exists(knowledge_dir):
        return ["Dummy knowledge base if directory not found."]
    for filename in os.listdir(knowledge_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(knowledge_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split by newlines into chunks
                chunks = [c.strip() for c in content.split('\n') if len(c.strip()) > 10]
                docs.extend(chunks)
    return docs if docs else ["Dummy knowledge base."]

WIKI_DOCS = load_corpus()

def get_retriever(mode='hybrid'):
    if mode == 'tfidf':
        return TFIDFRetriever(WIKI_DOCS)
    elif mode == 'faiss':
        return FAISSRetriever(WIKI_DOCS)
    else:
        return HybridRetriever(WIKI_DOCS)

# Global singleton
_retriever = None

def search_knowledge_base(query, mode='hybrid'):
    global _retriever
    if _retriever is None:
        try:
            _retriever = get_retriever(mode)
        except Exception as e:
            return f"[RAG Engine Initialization Failed: {str(e)}]"
            
    results = _retriever.search(query, top_k=1)
    if not results:
        return "No relevant information found in the knowledge base."
        
    # results contains (doc, score)
    best_doc = results[0][0]
    return f"Retrieved Context: {best_doc}"
