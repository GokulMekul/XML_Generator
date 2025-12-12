import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import pickle

PDF_FILE = "Learn.pdf"
INDEX_PATH = "index.faiss"
CHUNKS_PATH = "chunks.pkl"

# -----------------------------
# 1. Load PDF and chunk
# -----------------------------
def split_chunks(text, size=1000, overlap=100):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    return chunks

dataset = []
reader = PdfReader(PDF_FILE)
for page in reader.pages:
    text = page.extract_text()
    if text:
        dataset.extend(split_chunks(text))

print("Total Chunks:", len(dataset))

# -----------------------------
# 2. Embedding model
# -----------------------------
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
emb = model.encode(dataset, normalize_embeddings=True).astype("float32")

# -----------------------------
# 3. Build FAISS Index
# -----------------------------
dimension = emb.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(emb)

# -----------------------------
# 4. Save Index + Chunks
# -----------------------------
faiss.write_index(index, INDEX_PATH)

with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(dataset, f)

print("Index + chunks saved successfully!")
