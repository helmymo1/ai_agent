import fitz  # PyMuPDF
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

MODEL = None

def get_model():
    """Loads the sentence transformer model."""
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return MODEL

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text from a given PDF file."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

def chunk_text(text: str, chunk_size: int = 1024, chunk_overlap: int = 128) -> list[str]:
    """
    Splits a text into overlapping chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def search(query: str, chunks: list[str], embeddings: np.ndarray, top_k: int = 2):
    """
    Finds the most relevant chunks for a given query.
    """
    model = get_model()

    # 1. Embed the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # 2. Calculate cosine similarity between the query and all chunks
    similarities = util.cos_sim(query_embedding, embeddings)[0]

    # 3. Find the top_k most similar chunks using torch.topk
    top_k_results = torch.topk(similarities, k=min(top_k, len(chunks)))

    # 4. Return the results
    results = []
    for score, idx in zip(top_k_results.values, top_k_results.indices):
        results.append({
            "chunk": chunks[idx],
            "similarity": float(score)
        })
    return results
