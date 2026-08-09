import re

def extract_keywords(text):
    # Basic normalization and tokenization
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    # Stopwords filter
    stopwords = {"what", "is", "my", "the", "a", "an", "am", "i", "do", "does", "to", "in", "on", "of", "for", "are"}
    return [w for w in words if w not in stopwords]

def retrieve(query, long_term_memory, top_k=1):
    query_keywords = set(extract_keywords(query))
    if not query_keywords:
        return []

    scored_facts = []
    for fact in long_term_memory.get_all_facts():
        fact_text = f"{fact['key']} {fact['value']}".lower()
        fact_keywords = set(extract_keywords(fact_text))
        
        # Calculate overlap score
        overlap = len(query_keywords.intersection(fact_keywords))
        if overlap > 0:
            scored_facts.append((overlap, fact))
            
    # Sort by overlap score descending
    scored_facts.sort(key=lambda x: x[0], reverse=True)
    return [fact for score, fact in scored_facts[:top_k]]
