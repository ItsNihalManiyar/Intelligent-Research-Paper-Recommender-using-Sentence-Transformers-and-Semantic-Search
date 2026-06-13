import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest
import faiss

# Add src directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding.generate_embeddings import generate_embeddings
from src.search.faiss_index import build_faiss_index, load_faiss_index
from src.search.recommender import PaperRecommender

@pytest.fixture
def mock_pipeline_data():
    """
    Creates temporary mock dataset, embeddings, and FAISS index files for pipeline test.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a mock metadata CSV
        mock_data = {
            'paper_id': ['1', '2', '3'],
            'title': ['Deep learning crop yield', 'Machine learning and deep neural networks', 'FastAPI backend development'],
            'abstract': [
                'We predict agricultural crop yield using deep learning models on satellite imagery.',
                'A study on machine learning algorithms and neural network architectures.',
                'Building high performance APIs with Python FastAPI and microservices.'
            ],
            'authors': ['Author A', 'Author B', 'Author C'],
            'year': [2022, 2021, 2023],
            'category': ['cs.LG', 'cs.LG', 'cs.SE']
        }
        df = pd.DataFrame(mock_data)
        csv_path = os.path.join(tmp_dir, 'cleaned_papers.csv')
        df.to_csv(csv_path, index=False)
        
        # Paths for output files
        embeddings_path = os.path.join(tmp_dir, 'paper_embeddings.npy')
        index_path = os.path.join(tmp_dir, 'faiss_index.bin')
        
        yield {
            'csv_path': csv_path,
            'embeddings_path': embeddings_path,
            'index_path': index_path,
            'temp_dir': tmp_dir
        }

def test_pipeline_integration(mock_pipeline_data):
    """
    Tests embedding generation, FAISS indexing, and recommender search retrieval.
    """
    csv_path = mock_pipeline_data['csv_path']
    embeddings_path = mock_pipeline_data['embeddings_path']
    index_path = mock_pipeline_data['index_path']
    
    # 1. Test Embedding Generation
    generate_embeddings(
        input_path=csv_path,
        output_path=embeddings_path,
        model_name="all-MiniLM-L6-v2"
    )
    
    assert os.path.exists(embeddings_path), "Embeddings file was not created"
    embeddings = np.load(embeddings_path)
    assert embeddings.shape[0] == 3, "Embedding row count does not match dataset size"
    # MiniLM-L6-v2 has dimension 384
    assert embeddings.shape[1] == 384, "Embedding dimensions are incorrect"
    
    # 2. Test FAISS Index Building and Loading
    index = build_faiss_index(
        embeddings_path=embeddings_path,
        index_output_path=index_path
    )
    
    assert os.path.exists(index_path), "FAISS index file was not created"
    assert index.ntotal == 3, "Index count does not match row count"
    
    loaded_index = load_faiss_index(index_path)
    assert loaded_index.ntotal == 3
    
    # 3. Test Recommendation Output Structure
    recommender = PaperRecommender(
        metadata_path=csv_path,
        index_path=index_path,
        model_name="all-MiniLM-L6-v2"
    )
    
    query = "deep learning neural network"
    results = recommender.search(query=query, top_k=2)
    
    # Check that recommendation output size respects top_k
    assert len(results) <= 2, "Returned more results than top_k"
    assert len(results) > 0, "No results returned"
    
    # Check item schema
    for item in results:
        assert 'paper_id' in item
        assert 'title' in item
        assert 'abstract' in item
        assert 'authors' in item
        assert 'year' in item
        assert 'category' in item
        assert 'similarity_score' in item
        
        # Check type/values of similarity scores
        assert isinstance(item['similarity_score'], float)
        assert 0.0 <= item['similarity_score'] <= 1.0
        
    # The first result should be one of the deep learning ones
    first_title = results[0]['title'].lower()
    assert 'deep' in first_title or 'neural' in first_title or 'crop' in first_title
