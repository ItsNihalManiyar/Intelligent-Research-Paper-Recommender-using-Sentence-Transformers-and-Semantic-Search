import os
import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Add src directory to python path to import modules properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.load_data import load_dataset

def generate_embeddings(input_path: str, output_path: str, model_name: str = "all-MiniLM-L6-v2") -> None:
    """
    Generates embeddings for research papers by combining title, category, and abstract.
    """
    df = load_dataset(input_path)
    
    print("Preparing texts for embedding...")
    # Concatenate title, category, and abstract
    text_inputs = (
        df['title'].fillna('') + " " + 
        df['category'].fillna('') + " " + 
        df['abstract'].fillna('')
    ).tolist()
    
    print(f"Loading SentenceTransformer model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print(f"Generating embeddings for {len(text_inputs)} papers...")
    # Show progress bar if possible or print message
    embeddings = model.encode(text_inputs, show_progress_bar=True, batch_size=64)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save embeddings to disk
    np.save(output_path, embeddings)
    print(f"Successfully generated and saved embeddings of shape {embeddings.shape} to {output_path}")

if __name__ == "__main__":
    processed_path = "data/processed/cleaned_papers.csv"
    embeddings_path = "embeddings/paper_embeddings.npy"
    if os.path.exists(processed_path):
        generate_embeddings(processed_path, embeddings_path)
    else:
        print(f"Cleaned dataset not found at {processed_path}. Cannot generate embeddings.")
