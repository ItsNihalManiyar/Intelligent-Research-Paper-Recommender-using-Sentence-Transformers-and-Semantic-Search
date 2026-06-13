import os
import sys
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Add src directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.recommender import PaperRecommender
from database.sqlite_manager import SQLiteManager
from utils.pdf_utils import extract_text_from_pdf
from utils.keyword_utils import extract_keywords

app = FastAPI(
    title="Intelligent Research Paper Recommender API",
    description="Vector search & recommendation API for academic papers using Sentence Transformers, FAISS, KeyBERT, and SQLite.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize modules
# Using fallback configurations in case data is not downloaded yet
try:
    recommender = PaperRecommender(
        metadata_path="data/processed/cleaned_papers.csv",
        index_path="models/faiss_index.bin"
    )
except Exception as e:
    print(f"Warning: Recommender system failed to initialize (probably because dataset is not downloaded/built yet): {e}")
    recommender = None

db = SQLiteManager("database/favorites.db")

# Pydantic Schemas
class SearchQuery(BaseModel):
    query: str
    top_k: int = 10
    category: Optional[str] = None
    year_start: Optional[int] = None
    year_end: Optional[int] = None

class FavoritePaper(BaseModel):
    paper_id: str
    title: str
    year: int

class RemoveFavoriteRequest(BaseModel):
    paper_id: str

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Intelligent Research Paper Recommender API!",
        "status": "ready" if recommender is not None else "waiting_for_data_pipeline",
        "api_endpoints": [
            "GET /",
            "POST /search",
            "POST /upload-pdf",
            "POST /favorite",
            "GET /favorites",
            "POST /remove-favorite"
        ]
    }

@app.post("/search")
def search_papers(payload: SearchQuery):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender model/index is not ready. Please run data pipeline first.")
    
    try:
        # Get recommendations
        recommendations = recommender.search(
            query=payload.query,
            top_k=payload.top_k,
            category_filter=payload.category,
            year_start=payload.year_start,
            year_end=payload.year_end
        )
        
        # Extract keywords for the query
        query_keywords = extract_keywords(payload.query, top_n=5)
        
        # Enrich recommendations with paper abstract keywords
        enriched_results = []
        for paper in recommendations:
            paper_keywords = extract_keywords(paper['abstract'], top_n=3)
            enriched_paper = {**paper, "keywords": paper_keywords}
            enriched_results.append(enriched_paper)
            
        return {
            "query": payload.query,
            "query_keywords": query_keywords,
            "results": enriched_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/upload-pdf")
def upload_pdf(
    file: UploadFile = File(...),
    top_k: int = Form(10),
    category: Optional[str] = Form(None),
    year_start: Optional[int] = Form(None),
    year_end: Optional[int] = Form(None)
):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender model/index is not ready. Please run data pipeline first.")
    
    try:
        # Read uploaded PDF file bytes
        file_bytes = file.file.read()
        
        # Extract text from the PDF
        extracted_text = extract_text_from_pdf(file_bytes, max_chars=1500)
        if not extracted_text:
            raise HTTPException(status_code=400, detail="Failed to extract readable text from the uploaded PDF.")
            
        # Search using extracted text as query
        recommendations = recommender.search(
            query=extracted_text,
            top_k=top_k,
            category_filter=category,
            year_start=year_start,
            year_end=year_end
        )
        
        # Enrich results with keywords
        enriched_results = []
        for paper in recommendations:
            paper_keywords = extract_keywords(paper['abstract'], top_n=3)
            enriched_paper = {**paper, "keywords": paper_keywords}
            enriched_results.append(enriched_paper)
            
        return {
            "extracted_text_preview": extracted_text[:300] + "...",
            "results": enriched_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF recommendation failed: {str(e)}")

@app.post("/favorite")
def favorite_paper(payload: FavoritePaper):
    success = db.add_favorite(
        paper_id=payload.paper_id,
        title=payload.title,
        year=payload.year
    )
    if not success:
        return {"message": "Paper already in favorites or failed to add", "status": "no-op"}
    return {"message": "Paper successfully added to favorites", "status": "success"}

@app.get("/favorites")
def get_favorites():
    favorites_list = db.get_favorites()
    return {"favorites": favorites_list}

@app.post("/remove-favorite")
def remove_favorite(payload: RemoveFavoriteRequest):
    success = db.remove_favorite(paper_id=payload.paper_id)
    if not success:
        raise HTTPException(status_code=404, detail="Paper not found in favorites.")
    return {"message": "Paper successfully removed from favorites", "status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
