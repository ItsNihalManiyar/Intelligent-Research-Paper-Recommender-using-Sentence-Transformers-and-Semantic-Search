import os
import pandas as pd
from load_data import load_dataset

def clean_dataset(input_path: str, output_path: str) -> None:
    """
    Cleans the raw academic papers dataset according to specified criteria.
    """
    df = load_dataset(input_path)
    initial_count = len(df)
    
    # 1. Remove duplicates by paper_id
    df = df.drop_duplicates(subset=['paper_id'])
    after_dup_id = len(df)
    
    # 2. Remove null titles
    df = df.dropna(subset=['title'])
    after_null_title = len(df)
    
    # 3. Remove null abstracts
    df = df.dropna(subset=['abstract'])
    after_null_abstract = len(df)
    
    # 4. Remove abstracts under 100 characters
    df = df[df['abstract'].str.strip().str.len() >= 100]
    final_count = len(df)
    
    # Log cleaning metrics
    print(f"Data cleaning complete:")
    print(f"  Initial records: {initial_count}")
    print(f"  Removed duplicates by paper_id: {initial_count - after_dup_id}")
    print(f"  Removed null titles: {after_dup_id - after_null_title}")
    print(f"  Removed null abstracts: {after_null_title - after_null_abstract}")
    print(f"  Removed short abstracts (<100 chars): {after_null_abstract - final_count}")
    print(f"  Final cleaned records: {final_count}")
    
    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset to {output_path}")

if __name__ == "__main__":
    raw_path = "data/raw/arxiv_raw.csv"
    processed_path = "data/processed/cleaned_papers.csv"
    if os.path.exists(raw_path):
        clean_dataset(raw_path, processed_path)
    else:
        print(f"Raw dataset not found at {raw_path}. Cannot clean.")
