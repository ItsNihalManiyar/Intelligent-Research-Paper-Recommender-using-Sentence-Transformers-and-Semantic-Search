# Intelligent Research Paper Recommender

An academic research paper recommendation system that allows users to enter a natural language query and returns the most relevant research papers using semantic search. Powered by Sentence Transformers, FAISS vector indexing, KeyBERT keyword extraction, and SQLite favorites management, with both a FastAPI backend and a Streamlit web interface.

---

## Architecture

The system follows a modular architecture:

1. **Dataset Pipeline**: Fetches real arXiv research paper metadata, cleans entries, validates schemas, and outputs processed datasets.
2. **Embedding Pipeline**: Computes high-dimensional dense vector embeddings using the `all-MiniLM-L6-v2` SentenceTransformer model.
3. **Vector Similarity Index**: Indexing and fast nearest-neighbor retrieval using FAISS (`IndexFlatL2`).
4. **Relational Storage**: SQLite database storing user-defined favorite papers.
5. **Backend REST API**: FastAPI server hosting recommendation and database integration endpoints.
6. **Frontend Web App**: Streamlit client consuming the FastAPI endpoints with fallback options to direct python module imports.

---

## Project Structure

```
project-root/
├── data/
│   ├── raw/
│   ├── processed/
│   └── reports/
├── embeddings/
├── models/
├── database/
├── docs/
├── src/
│   ├── data_processing/
│   │   ├── download_dataset.py
│   │   ├── load_data.py
│   │   ├── profile_data.py
│   │   ├── clean_data.py
│   │   └── validate_data.py
│   ├── embedding/
│   │   └── generate_embeddings.py
│   ├── search/
│   │   ├── faiss_index.py
│   │   └── recommender.py
│   ├── database/
│   │   └── sqlite_manager.py
│   ├── utils/
│   │   ├── pdf_utils.py
│   │   └── keyword_utils.py
│   └── api/
│       └── main.py
├── app/
│   └── streamlit/
│       └── app.py
├── tests/
│   └── test_recommender.py
├── requirements.txt
└── README.md
```

---

## Installation

1. Clone or navigate into the project directory:
   ```bash
   cd "Intelligent-Research-Paper-Recommender-using-Sentence-Transformers-and-Semantic-Search"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset Preparation & Embedding Pipeline

To download, process, validate the dataset, and build the FAISS vector index, run the following steps in sequence:

1. **Download metadata**: Downloads 5,000 recent Machine Learning/AI research papers from arXiv API.
   ```bash
   python src/data_processing/download_dataset.py
   ```

2. **Profile raw dataset**: Creates a summary statistical report at `data/reports/data_profile_report.md`.
   ```bash
   python src/data_processing/profile_data.py
   ```

3. **Clean dataset**: Cleans records and filters out short abstracts, saving to `data/processed/cleaned_papers.csv`.
   ```bash
   python src/data_processing/clean_data.py
   ```

4. **Validate dataset**: Validates schemas and quality, producing `data/reports/data_validation_report.md`.
   ```bash
   python src/data_processing/validate_data.py
   ```

5. **Generate embeddings**: Encodes `title + category + abstract` using SentenceTransformers.
   ```bash
   python src/embedding/generate_embeddings.py
   ```

6. **Build FAISS index**: Stores embeddings into a Flat L2 FAISS index binary at `models/faiss_index.bin`.
   ```bash
   python src/search/faiss_index.py
   ```

---

## Running the Application

### 1. Start the FastAPI Backend
Start the REST API server:
```bash
python src/api/main.py
```
Or use Uvicorn directly:
```bash
uvicorn src.api.main:app --reload
```
API Documentation will be accessible at: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Start the Streamlit Frontend
Launch the interactive web interface:
```bash
streamlit run app/streamlit/app.py
```
The application will open in your default browser at: [http://localhost:8501](http://localhost:8501)

---

## Running Tests

Run unit/integration tests with:
```bash
pytest tests/
```

---

## Future Improvements

* Add support for real-time arXiv category feed indexing.
* Upgrade vector search index with HNSW (Hierarchical Navigable Small World) for sub-millisecond retrieval on larger datasets.
* Support multi-page PDF processing with semantic chunking to handle full papers.
* Implement user query history profiles using SQLite.

---

## Screenshots

*(Upload and add screenshots of Streamlit interface cards, similarity score progress bars, and the favorites section)*
