# utils.py
import hashlib
import time

def check_cache(question, embed_model, cache_db, threshold=0.90):
    """Checks the semantic cache for a similar question."""
    results = cache_db.similarity_search_with_relevance_scores(question, k=1)
    
    if results and results[0][1] >= threshold:
        doc, score = results[0]
        # Use your safe-get fix
        return doc.metadata.get("answer"), score
            
    return None, 0

def save_to_cache(question, answer, cache_db):
    """Saves a new Q&A pair to the cache database."""
    # Generate unique ID based on the question hash
    q_id = hashlib.md5(question.lower().encode()).hexdigest()
    
    cache_db.add_texts(
        texts=[question],
        ids=[q_id],
        metadatas=[{"answer": answer, "timestamp": time.time()}]
    )

def cleanup_cache_by_id(cache_db, max_size=50):
    """Keeps the cache under a certain size limit (LRU logic)."""
    data = cache_db.get(include=['metadatas'])
    ids = data['ids']
    metadatas = data['metadatas']
    
    if len(ids) > max_size:
        # Sort by timestamp (oldest first)
        entries = []
        for i in range(len(ids)):
            ts = metadatas[i].get("timestamp", 0)
            entries.append((ids[i], ts))
        
        entries.sort(key=lambda x: x[1])
        
        # Determine how many to delete
        num_to_delete = len(ids) - max_size
        ids_to_remove = [e[0] for e in entries[:num_to_delete]]
        
        cache_db.delete(ids=ids_to_remove)