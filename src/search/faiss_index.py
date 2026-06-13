import os
import numpy as np
import faiss
from typing import Tuple

def build_faiss_index(embeddings_path: str, index_output_path: str) -> faiss.Index:
    """
    Builds a FAISS IndexFlatL2 from the saved numpy embeddings and saves it.
    """
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found at: {embeddings_path}")
        
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path).astype('float32')
    
    dimension = embeddings.shape[1]
    print(f"Embedding dimension: {dimension}")
    
    # Create FAISS Flat L2 index
    index = faiss.IndexFlatL2(dimension)
    
    print("Adding embeddings to FAISS index...")
    index.add(embeddings)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(index_output_path), exist_ok=True)
    
    # Persist the index
    faiss.write_index(index, index_output_path)
    print(f"Successfully built and saved FAISS index to {index_output_path} with {index.ntotal} vectors.")
    return index

def load_faiss_index(index_path: str) -> faiss.Index:
    """
    Loads a persisted FAISS index.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file not found at: {index_path}")
        
    print(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(index_path)
    print(f"Loaded FAISS index with {index.ntotal} vectors.")
    return index

if __name__ == "__main__":
    emb_path = "embeddings/paper_embeddings.npy"
    idx_path = "models/faiss_index.bin"
    if os.path.exists(emb_path):
        build_faiss_index(emb_path, idx_path)
    else:
        print(f"Embeddings not found at {emb_path}. Cannot build FAISS index.")
