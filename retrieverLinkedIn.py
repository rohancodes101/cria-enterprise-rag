import faiss
import pickle
import numpy as np
import os
from llama_cpp import Llama 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "embeddings", "granite_cr_index.bin")
META_PATH = os.path.join(BASE_DIR, "embeddings", "granite_cr_metadata.pkl")

# Direct path to your model
EMBED_MODEL = os.path.abspath(os.path.join(BASE_DIR, "../models/granite-embedding-278m-multilingual-Q4_K_M.gguf"))

# =====================================================
# INITIALIZE LLM ENGINE
# =====================================================
print("Initializing Embedding Engine...")
llm = Llama(model_path=EMBED_MODEL, embedding=True, verbose=False)

# =====================================================
# LOAD FAISS & METADATA
# =====================================================
print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

# =====================================================
# CLEAN EMBEDDING FUNCTION
# =====================================================
def embed_query(text):
    # No more subprocess or regex!
    output = llm.create_embedding(text)
    return np.array(output['data'][0]['embedding'], dtype=np.float32)

# =====================================================
# SEARCH & CLI (Remains mostly the same)
# =====================================================
def search_crs(query, top_k=5):
    query_vec = embed_query(query).reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)
    
    results = []
    for rank, idx in enumerate(indices[0]):
        cr = metadata[idx]
        results.append({
            "rank": rank + 1,
            "cr_number": cr.get("number"),
            "short_description": cr.get("short_description"),
            "distance": float(distances[0][rank])
        })
    return results

while True:
    q = input("\nEnter change idea (exit to quit): ")
    if q.lower() == "exit": break
    
    res = search_crs(q)
    for r in res:
        print(f"\nRank {r['rank']} | CR: {r['cr_number']}")
        print(f"Short: {r['short_description']}")
        print(f"Distance: {r['distance']:.4f}")