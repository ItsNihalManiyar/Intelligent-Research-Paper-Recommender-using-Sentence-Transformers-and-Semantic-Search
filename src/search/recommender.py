import os
import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

# Add src directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.faiss_index import load_faiss_index

class PaperRecommender:
    def __init__(self, 
                 metadata_path: str = "data/processed/cleaned_papers.csv", 
                 index_path: str = "models/faiss_index.bin", 
                 model_name: str = "all-MiniLM-L6-v2"):
        self.metadata_path = metadata_path
        self.index_path = index_path
        self.model_name = model_name
        
        self.df = None
        self.index = None
        self.model = None
        
        self.load_resources()
        
    def load_resources(self) -> None:
        """
        Loads metadata, FAISS index, and SentenceTransformer model.
        """
        if os.path.exists(self.metadata_path):
            self.df = pd.read_csv(self.metadata_path)
        else:
            print(f"Warning: Metadata file not found at {self.metadata_path}")
            
        if os.path.exists(self.index_path):
            self.index = load_faiss_index(self.index_path)
        else:
            print(f"Warning: FAISS index file not found at {self.index_path}")
            
        print(f"Loading SentenceTransformer model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        
    def search(self, 
               query: str, 
               top_k: int = 10, 
               category_filter: Optional[str] = None, 
               year_start: Optional[int] = None, 
               year_end: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Searches for papers similar to the query, applying filters if provided.
        """
        if self.df is None or self.index is None or self.model is None:
            raise ValueError("Recommender resources not fully loaded. Check metadata, model, and index paths.")
            
        # Encode query
        query_vector = self.model.encode([query]).astype('float32')
        
        # If no filters, we can just search top_k directly
        # If there are filters, search more candidates to ensure we have top_k after filtering
        search_k = min(len(self.df), top_k * 10 if (category_filter or year_start or year_end) else top_k)
        
        distances, indices = self.index.search(query_vector, search_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.df):
                continue
                
            row = self.df.iloc[idx]
            
            # Apply category filter
            if category_filter and str(row['category']).strip().lower() != category_filter.strip().lower():
                continue
                
            # Apply year range filter
            if year_start is not None and (pd.isna(row['year']) or int(row['year']) < year_start):
                continue
            if year_end is not None and (pd.isna(row['year']) or int(row['year']) > year_end):
                continue
                
            # Compute similarity score from L2 distance
            # Since SentenceTransformer embeddings are L2 normalized, 
            # cosine similarity = 1 - (L2_distance^2)/2
            # Let's map it safely to [0, 1]
            similarity = float(1.0 - (dist / 2.0))
            similarity = max(0.0, min(1.0, similarity))
            
            results.append({
                'paper_id': str(row['paper_id']),
                'title': str(row['title']),
                'abstract': str(row['abstract']),
                'authors': str(row['authors']),
                'year': int(row['year']) if not pd.isna(row['year']) else None,
                'category': str(row['category']),
                'similarity_score': similarity
            })
            
            # Stop if we have gathered enough top_k results
            if len(results) >= top_k:
                break
                
        return results

if __name__ == "__main__":
    # Quick test
    try:
        recommender = PaperRecommender()
        query = "deep learning crop yield prediction"
        print(f"Searching for: '{query}'")
        res = recommender.search(query, top_k=3)
        for i, paper in enumerate(res):
            print(f"\nResult {i+1} [Score: {paper['similarity_score']:.4f}]:")
            print(f"Title: {paper['title']}")
            print(f"Category: {paper['category']} | Year: {paper['year']}")
    except Exception as e:
        print(f"Recommender test omitted/failed: {e}")
