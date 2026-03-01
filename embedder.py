import json
import faiss
import pickle
import numpy as np
import os
import sys
from llama_cpp import Llama
sys.stderr = open(os.devnull, 'w')

# =====================================================
# PATH CONFIG
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
DATA_PATH = os.path.join(BASE_DIR, "dataset.json")
INDEX_PATH = os.path.join(BASE_DIR, "embeddings", "granite_cr_index.bin")
META_PATH = os.path.join(BASE_DIR, "embeddings", "granite_cr_metadata.pkl")

EMBED_MODEL = os.path.abspath(
    os.path.join(BASE_DIR, "../models/granite-embedding-278m-multilingual-Q4_K_M.gguf")
)


# print("BASE_DIR:", BASE_DIR)
# print("DATA_PATH:", DATA_PATH)
# print("Exists?", os.path.exists(DATA_PATH))


# =====================================================
# LOAD EMBEDDING MODEL (ONCE)
# =====================================================

print("Loading embedding model...")

embedder = Llama(
    model_path=EMBED_MODEL,
    embedding=True,
    n_ctx=512,
    verbose=False
)

print("Embedding model loaded\n")

# =====================================================
# TEXT CLEANING
# =====================================================

def clean_text(x, max_chars=800):
    if not isinstance(x, str):
        return ""
    x = x.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return x[:max_chars]

# =====================================================
# BUILD CR TEXT
# =====================================================

def build_cr_text(cr):
    return f"""
Short Description:
{clean_text(cr.get('short_description',''),200)}

Description:
{clean_text(cr.get('description',''),600)}

Reason for Change:
{clean_text(cr.get('u_reason_for_change',''),400)}

Implementation Plan Summary:
{clean_text(cr.get('implementation_plan',''),600)}

Risk Impact:
{clean_text(cr.get('risk_impact_analysis',''),300)}

Test Plan Summary:
{clean_text(cr.get('test_plan',''),400)}

Backout Summary:
{clean_text(cr.get('backout_plan',''),300)}
""".strip()


# =====================================================
# LOAD DATASET
# =====================================================

print("Loading dataset...")

with open(DATA_PATH, "r", encoding="utf-8", errors="ignore") as f:
    data = json.load(f)

records = data.get("records", [])
print("Total CR records:", len(records))
print()

# =====================================================
# GENERATE EMBEDDINGS
# =====================================================

vectors = []
metadata = []

for i, cr in enumerate(records, start=1):

    print(f"Embedding CR {i}/{len(records)}")

    text = build_cr_text(cr)

    vec = embedder.embed(text)

    vectors.append(vec)
    metadata.append(cr)

vectors = np.array(vectors, dtype=np.float32)

print("\nFinal vector shape:", vectors.shape)

# =====================================================
# BUILD FAISS INDEX
# =====================================================

print("\nBuilding FAISS index...")

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

faiss.write_index(index, INDEX_PATH)

with open(META_PATH, "wb") as f:
    pickle.dump(metadata, f)

print("\nFAISS INDEX CREATED SUCCESSFULLY")
print("Vectors:", len(metadata))
print("Dimension:", vectors.shape[1])

sys.stderr = sys.__stderr__