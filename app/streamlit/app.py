import os
import sys
import requests
import streamlit as st
import pandas as pd
from typing import Dict, Any, List

# Add src to system path for direct import fallback
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Try to import directly in case FastAPI server is not running
from src.search.recommender import PaperRecommender
from src.database.sqlite_manager import SQLiteManager
from src.utils.pdf_utils import extract_text_from_pdf
from src.utils.keyword_utils import extract_keywords

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Intelligent Paper Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(45deg, #1f4068, #162447, #e43f5a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .paper-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid #1f4068;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .dark .paper-card {
        background-color: #1a1a24;
        border-left: 5px solid #e43f5a;
    }
    .paper-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1f4068;
        margin-bottom: 0.5rem;
    }
    .dark .paper-title {
        color: #e43f5a;
    }
    .metadata-tag {
        display: inline-block;
        background-color: #e8eaf6;
        color: #3f51b5;
        font-size: 0.8rem;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        margin-right: 0.5rem;
        font-weight: 600;
    }
    .keyword-tag {
        display: inline-block;
        background-color: #e0f2f1;
        color: #00796b;
        font-size: 0.75rem;
        padding: 0.15rem 0.5rem;
        border-radius: 5px;
        margin-right: 0.4rem;
        margin-top: 0.3rem;
    }
    .score-label {
        font-weight: 600;
        font-size: 0.9rem;
        color: #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Check API Connection -----------------
@st.cache_data(show_spinner=False)
def check_api_status() -> bool:
    try:
        r = requests.get(f"{API_URL}/", timeout=1)
        return r.status_code == 200
    except requests.RequestException:
        return False

api_online = check_api_status()

# ----------------- Initialize Local Fallback If Needed -----------------
@st.cache_resource(show_spinner=True)
def get_local_recommender():
    try:
        return PaperRecommender(
            metadata_path="data/processed/cleaned_papers.csv",
            index_path="models/faiss_index.bin"
        )
    except Exception as e:
        return None

local_recommender = None
local_db = SQLiteManager("database/favorites.db")

if not api_online:
    local_recommender = get_local_recommender()

# ----------------- Helper Actions -----------------
def fetch_favorites() -> List[Dict[str, Any]]:
    if api_online:
        try:
            r = requests.get(f"{API_URL}/favorites")
            return r.json().get("favorites", [])
        except Exception:
            pass
    return local_db.get_favorites()

def add_to_favorites(paper: Dict[str, Any]):
    if api_online:
        try:
            requests.post(f"{API_URL}/favorite", json={
                "paper_id": paper["paper_id"],
                "title": paper["title"],
                "year": paper["year"] or 0
            })
            return
        except Exception:
            pass
    local_db.add_favorite(paper["paper_id"], paper["title"], paper["year"] or 0)

def remove_from_favorites(paper_id: str):
    if api_online:
        try:
            requests.post(f"{API_URL}/remove-favorite", json={"paper_id": paper_id})
            return
        except Exception:
            pass
    local_db.remove_favorite(paper_id)

# ----------------- Load Metadata for Filters -----------------
@st.cache_data
def get_filter_options():
    # Attempt to load options from clean papers
    csv_path = "data/processed/cleaned_papers.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        categories = sorted(df['category'].dropna().unique().tolist())
        min_year = int(df['year'].min()) if not df['year'].isnull().all() else 2000
        max_year = int(df['year'].max()) if not df['year'].isnull().all() else 2026
        # to solve streamlit.errors.StreamlitAPIException: Slider `min_value` must be less than the `max_value`.
        if min_year == max_year:
            min_year = max_year - 1

        return categories, min_year, max_year
    return ["cs.LG", "cs.CV", "cs.CL", "cs.AI", "stat.ML"], 2018, 2026

categories, min_year_limit, max_year_limit = get_filter_options()

# ----------------- UI Layout -----------------
st.markdown("<div class='main-title'>Intelligent Research Paper Recommender</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Semantic research explorer powered by Sentence Transformers, FAISS vector search, and KeyBERT</div>", unsafe_allow_html=True)

# Status indicators in Sidebar
st.sidebar.title("⚙️ System Status")
if api_online:
    st.sidebar.success("FastAPI Backend: ONLINE")
else:
    st.sidebar.warning("FastAPI Backend: OFFLINE (Running Local Mode)")

if (api_online) or (local_recommender is not None):
    st.sidebar.info("FAISS Vector Store: READY")
else:
    st.sidebar.error("FAISS Vector Store: NOT FOUND (Run pipeline first)")

# ----------------- Sidebar Filters -----------------
st.sidebar.title("🎯 Query Filters")
category_filter = st.sidebar.selectbox("Category Filter", ["All"] + categories)
year_range = st.sidebar.slider(
    "Year Range",
    min_value=min_year_limit,
    max_value=max_year_limit,
    value=(min_year_limit, max_year_limit)
)

# Convert filters
cat_val = None if category_filter == "All" else category_filter
yr_start, yr_end = year_range

# ----------------- Main Interface Tabs -----------------
tab_search, tab_pdf, tab_favs = st.tabs(["Semantic Search", "Search via PDF", "Saved Favorites"])

# ----------------- Tab 1: Semantic Search -----------------
with tab_search:
    st.subheader("Enter your research interest:")
    query_input = st.text_input(
        "Search Query",
        placeholder="e.g., crop yield prediction using deep learning or graph neural networks for drug discovery",
        label_visibility="collapsed"
    )
    
    top_k = st.slider("Number of recommendations", min_value=3, max_value=20, value=10)
    search_button = st.button("Recommend Papers", type="primary")
    
    results = []
    query_keywords = []
    
    if search_button and query_input:
        with st.spinner("Searching and extracting keywords..."):
            if api_online:
                try:
                    payload = {
                        "query": query_input,
                        "top_k": top_k,
                        "category": cat_val,
                        "year_start": yr_start,
                        "year_end": yr_end
                    }
                    r = requests.post(f"{API_URL}/search", json=payload)
                    if r.status_code == 200:
                        data = r.json()
                        results = data.get("results", [])
                        query_keywords = data.get("query_keywords", [])
                except Exception as e:
                    st.error(f"Error calling FastAPI: {e}")
            
            # Fallback/Local execution
            if not results and local_recommender is not None:
                results = local_recommender.search(
                    query=query_input,
                    top_k=top_k,
                    category_filter=cat_val,
                    year_start=yr_start,
                    year_end=yr_end
                )
                query_keywords = extract_keywords(query_input, top_n=5)
                # Add abstract keywords locally
                for r_item in results:
                    r_item['keywords'] = extract_keywords(r_item['abstract'], top_n=3)

        # Display query keywords
        if query_keywords:
            st.write("**Extracted Search Keywords:**")
            kw_html = "".join([f"<span class='keyword-tag'>{kw}</span>" for kw in query_keywords])
            st.markdown(kw_html, unsafe_allow_html=True)
            st.write("")
            
        # Display results
        if results:
            st.subheader(f"Top {len(results)} Recommendations")
            for paper in results:
                with st.container():
                    st.markdown(f"""
                    <div class='paper-card'>
                        <div class='paper-title'>{paper['title']}</div>
                        <div>
                            <span class='metadata-tag'>🆔 {paper['paper_id']}</span>
                            <span class='metadata-tag'>📂 {paper['category']}</span>
                            <span class='metadata-tag'>📅 {paper['year']}</span>
                        </div>
                        <div style='margin-top: 0.5rem;'>
                            <strong>Authors:</strong> {paper['authors']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Similarity Score progress bar
                    col_score, col_button = st.columns([4, 1])
                    with col_score:
                        st.markdown(f"<span class='score-label'>Similarity Match Score: {paper['similarity_score']*100:.1f}%</span>", unsafe_allow_html=True)
                        st.progress(paper['similarity_score'])
                        
                    with col_button:
                        # Check if already in favorites to display appropriate button label
                        all_fav_ids = [f['paper_id'] for f in fetch_favorites()]
                        if paper['paper_id'] in all_fav_ids:
                            if st.button("Starred ⭐", key=f"fav_btn_s_{paper['paper_id']}", disabled=True):
                                pass
                        else:
                            if st.button("Favorite ⭐", key=f"fav_btn_{paper['paper_id']}"):
                                add_to_favorites(paper)
                                st.success("Added to favorites!")
                                st.rerun()
                                
                    # Abstract preview in Expander
                    with st.expander("Show Abstract & Paper Details"):
                        st.write(paper['abstract'])
                        if 'keywords' in paper and paper['keywords']:
                            st.write("**Paper Keywords:**")
                            tags = "".join([f"<span class='keyword-tag'>{kw}</span>" for kw in paper['keywords']])
                            st.markdown(tags, unsafe_allow_html=True)
                            
                    st.markdown("---")
        else:
            st.info("No matching papers found. Try adjusting your query or expanding filters.")

# ----------------- Tab 2: Search via PDF -----------------
with tab_pdf:
    st.subheader("Upload a Research Paper PDF to find similar papers:")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    top_k_pdf = st.slider("Number of recommendations (PDF)", min_value=3, max_value=20, value=10, key="pdf_k")
    
    pdf_results = []
    
    if uploaded_file is not None:
        if st.button("Analyze PDF and Recommend", type="primary"):
            with st.spinner("Extracting text from PDF and computing recommendations..."):
                if api_online:
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        data = {
                            "top_k": top_k_pdf,
                            "category": cat_val if cat_val else "",
                            "year_start": yr_start if yr_start else "",
                            "year_end": yr_end if yr_end else ""
                        }
                        r = requests.post(f"{API_URL}/upload-pdf", files=files, data=data)
                        if r.status_code == 200:
                            pdf_results = r.json().get("results", [])
                    except Exception as e:
                        st.error(f"Error calling FastAPI PDF endpoint: {e}")
                        
                # Fallback
                if not pdf_results and local_recommender is not None:
                    extracted_text = extract_text_from_pdf(uploaded_file.getvalue(), max_chars=1500)
                    if extracted_text:
                        st.info(f"Extracted Text Preview: {extracted_text[:200]}...")
                        pdf_results = local_recommender.search(
                            query=extracted_text,
                            top_k=top_k_pdf,
                            category_filter=cat_val,
                            year_start=yr_start,
                            year_end=yr_end
                        )
                        for r_item in pdf_results:
                            r_item['keywords'] = extract_keywords(r_item['abstract'], top_n=3)
                    else:
                        st.error("Could not extract readable text from PDF.")
            
            # Display PDF results
            if pdf_results:
                st.subheader("Papers similar to uploaded PDF:")
                for paper in pdf_results:
                    with st.container():
                        st.markdown(f"""
                        <div class='paper-card'>
                            <div class='paper-title'>{paper['title']}</div>
                            <div>
                                <span class='metadata-tag'>🆔 {paper['paper_id']}</span>
                                <span class='metadata-tag'>📂 {paper['category']}</span>
                                <span class='metadata-tag'>📅 {paper['year']}</span>
                            </div>
                            <div style='margin-top: 0.5rem;'>
                                <strong>Authors:</strong> {paper['authors']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_score, col_button = st.columns([4, 1])
                        with col_score:
                            st.markdown(f"<span class='score-label'>Similarity Match Score: {paper['similarity_score']*100:.1f}%</span>", unsafe_allow_html=True)
                            st.progress(paper['similarity_score'])
                            
                        with col_button:
                            all_fav_ids = [f['paper_id'] for f in fetch_favorites()]
                            if paper['paper_id'] in all_fav_ids:
                                if st.button("Starred ⭐", key=f"fav_btn_pdf_s_{paper['paper_id']}", disabled=True):
                                    pass
                            else:
                                if st.button("Favorite ⭐", key=f"fav_btn_pdf_{paper['paper_id']}"):
                                    add_to_favorites(paper)
                                    st.success("Added to favorites!")
                                    st.rerun()
                                    
                        with st.expander("Show Abstract"):
                            st.write(paper['abstract'])
                            if 'keywords' in paper and paper['keywords']:
                                st.write("**Paper Keywords:**")
                                tags = "".join([f"<span class='keyword-tag'>{kw}</span>" for kw in paper['keywords']])
                                st.markdown(tags, unsafe_allow_html=True)
                        st.markdown("---")
            else:
                st.info("No matching papers found.")

# ----------------- Tab 3: Saved Favorites -----------------
with tab_favs:
    st.subheader("Your Favorite Papers")
    fav_list = fetch_favorites()
    
    if fav_list:
        # Create a pandas DataFrame for nicer overview
        fav_df = pd.DataFrame(fav_list)
        # Drop sqlite internal primary key for user presentation
        if 'id' in fav_df.columns:
            fav_df = fav_df.drop(columns=['id'])
            
        st.dataframe(fav_df, use_container_width=True)
        
        # Display list with remove buttons
        for f_paper in fav_list:
            col_info, col_action = st.columns([5, 1])
            with col_info:
                st.markdown(f"**{f_paper['title']}** ({f_paper['year']})  \n`ID: {f_paper['paper_id']}`")
            with col_action:
                if st.button("Remove ❌", key=f"rem_fav_{f_paper['paper_id']}"):
                    remove_from_favorites(f_paper['paper_id'])
                    st.success("Removed!")
                    st.rerun()
            st.markdown("---")
    else:
        st.info("No papers added to favorites yet. Click the 'Favorite' button on recommendations.")
