import os
import sys
import requests
import streamlit as st
import pandas as pd
from typing import Dict, Any, List

# setup path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.search.recommender import PaperRecommender
from src.database.sqlite_manager import SQLiteManager
from src.utils.pdf_utils import extract_text_from_pdf
from src.utils.keyword_utils import extract_keywords

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Paper Recommender System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# UI styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 2rem;
    }
    .paper-card {
        background-color: #f9f9f9;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        border-left: 5px solid #1f4068;
    }
    .paper-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1f4068;
        margin-bottom: 0.4rem;
    }
    .metadata-tag {
        display: inline-block;
        background-color: #e8eaf6;
        color: #3f51b5;
        font-size: 0.8rem;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        margin-right: 0.4rem;
    }
    .keyword-tag {
        display: inline-block;
        background-color: #e0f2f1;
        color: #00796b;
        font-size: 0.75rem;
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
        margin-right: 0.3rem;
        margin-top: 0.3rem;
    }
    .score-label {
        font-weight: 500;
        font-size: 0.9rem;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# check if backend is running
@st.cache_data(show_spinner=False)
def check_api_status() -> bool:
    try:
        r = requests.get(f"{API_URL}/", timeout=1)
        return r.status_code == 200
    except requests.RequestException:
        return False

api_online = check_api_status()

# local loading logic if backend fails
@st.cache_resource(show_spinner=True)
def get_local_recommender():
    try:
        return PaperRecommender(
            metadata_path="data/processed/cleaned_papers.csv",
            index_path="models/faiss_index.bin"
        )
    except Exception:
        return None

local_recommender = None
local_db = SQLiteManager("database/favorites.db")

if not api_online:
    local_recommender = get_local_recommender()

# DB/API handlers for favorites tab
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

# load filtering data dynamically from dataset
@st.cache_data
def get_filter_options():
    csv_path = "data/processed/cleaned_papers.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        categories = sorted(df['category'].dropna().unique().tolist())
        min_year = int(df['year'].min()) if not df['year'].isnull().all() else 2000
        max_year = int(df['year'].max()) if not df['year'].isnull().all() else 2026
        
        # handles streamlit slider constraint where values can't match
        if min_year == max_year:
            min_year = max_year - 1

        return categories, min_year, max_year
    return ["cs.LG", "cs.CV", "cs.CL", "cs.AI", "stat.ML"], 2018, 2026

categories, min_year_limit, max_year_limit = get_filter_options()

# app layout interface
st.markdown("<div class='main-title'>Research Paper Recommender System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Search engine using NLP embeddings and FAISS similarity lookup</div>", unsafe_allow_html=True)

# sidebar dashboard settings
st.sidebar.title("System Monitor")
if api_online:
    st.sidebar.success("Backend: ONLINE")
else:
    st.sidebar.warning("Backend: OFFLINE (Using local fallback)")

if (api_online) or (local_recommender is not None):
    st.sidebar.info("FAISS Index: ACTIVE")
else:
    st.sidebar.error("FAISS Index: MISSING (Run local pipeline)")

st.sidebar.title("Search Modifiers")
category_filter = st.sidebar.selectbox("Filter by Category", ["All"] + categories)
year_range = st.sidebar.slider(
    "Filter by Publication Year",
    min_value=min_year_limit,
    max_value=max_year_limit,
    value=(min_year_limit, max_year_limit)
)

cat_val = None if category_filter == "All" else category_filter
yr_start, yr_end = year_range

# user interface tabs
tab_search, tab_pdf, tab_favs = st.tabs(["Text Query Search", "PDF Document Upload", "Saved Bookmarks"])

# TAB 1: TEXT QUERY SEARCH
with tab_search:
    st.subheader("What topic are you researching?")
    query_input = st.text_input(
        "Search Query",
        placeholder="Type keywords or research interest sentence here...",
        label_visibility="collapsed"
    )
    
    top_k = st.slider("Max results to fetch", min_value=3, max_value=20, value=10)
    search_button = st.button("Find Matching Papers", type="primary")
    
    results = []
    query_keywords = []
    
    if search_button and query_input:
        with st.spinner("Processing text weights..."):
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
                    st.error(f"API Error: {e}")
            
            # run local script computation if backend down
            if not results and local_recommender is not None:
                results = local_recommender.search(
                    query=query_input,
                    top_k=top_k,
                    category_filter=cat_val,
                    year_start=yr_start,
                    year_end=yr_end
                )
                query_keywords = extract_keywords(query_input, top_n=5)
                for r_item in results:
                    r_item['keywords'] = extract_keywords(r_item['abstract'], top_n=3)

        # display keywords extracted from search block
        if query_keywords:
            st.write("**Extracted Key Concepts:**")
            kw_html = "".join([f"<span class='keyword-tag'>{kw}</span>" for kw in query_keywords])
            st.markdown(kw_html, unsafe_allow_html=True)
            st.write("")
            
        # render target recommendations cards
        if results:
            st.subheader("Recommendations Results")
            for paper in results:
                with st.container():
                    st.markdown(f"""
                    <div class='paper-card'>
                        <div class='paper-title'>{paper['title']}</div>
                        <div>
                            <span class='metadata-tag'>ID: {paper['paper_id']}</span>
                            <span class='metadata-tag'>Class: {paper['category']}</span>
                            <span class='metadata-tag'>Year: {paper['year']}</span>
                        </div>
                        <div style='margin-top: 0.4rem; font-size: 0.9rem;'>
                            <strong>Authors:</strong> {paper['authors']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_score, col_button = st.columns([4, 1])
                    with col_score:
                        st.markdown(f"<span class='score-label'>Match Score: {paper['similarity_score']*100:.1f}%</span>", unsafe_allow_html=True)
                        st.progress(paper['similarity_score'])
                        
                    with col_button:
                        all_fav_ids = [f['paper_id'] for f in fetch_favorites()]
                        if paper['paper_id'] in all_fav_ids:
                            st.button("Saved ✓", key=f"fav_btn_s_{paper['paper_id']}", disabled=True)
                        else:
                            if st.button("Bookmark", key=f"fav_btn_{paper['paper_id']}"):
                                add_to_favorites(paper)
                                st.success("Saved!")
                                st.rerun()
                                
                    with st.expander("View Abstract & Details"):
                        st.write(paper['abstract'])
                        if 'keywords' in paper and paper['keywords']:
                            st.write("**Paper Keywords:**")
                            tags = "".join([f"<span class='keyword-tag'>{kw}</span>" for kw in paper['keywords']])
                            st.markdown(tags, unsafe_allow_html=True)
                            
                    st.markdown("---")
        else:
            st.info("No documents match current filters.")

# TAB 2: SEARCH VIA PDF UPLOAD
with tab_pdf:
    st.subheader("Upload a text-based PDF to locate related work:")
    uploaded_file = st.file_uploader("Select a file (.pdf)", type=["pdf"])
    top_k_pdf = st.slider("Max results to fetch", min_value=3, max_value=20, value=10, key="pdf_k")
    
    pdf_results = []
    
    if uploaded_file is not None:
        if st.button("Run PDF Analysis", type="primary"):
            with st.spinner("Parsing document structure..."):
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
                        st.error(f"Server endpoint execution error: {e}")
                        
                if not pdf_results and local_recommender is not None:
                    extracted_text = extract_text_from_pdf(uploaded_file.getvalue(), max_chars=1500)
                    if extracted_text:
                        st.info(f"Parsed Sample: {extracted_text[:150]}...")
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
                        st.error("Could not parse text blocks inside this file format.")
            
            if pdf_results:
                st.subheader("Calculated Similar Matches:")
                for paper in pdf_results:
                    with st.container():
                        st.markdown(f"""
                        <div class='paper-card'>
                            <div class='paper-title'>{paper['title']}</div>
                            <div>
                                <span class='metadata-tag'>ID: {paper['paper_id']}</span>
                                <span class='metadata-tag'>Class: {paper['category']}</span>
                                <span class='metadata-tag'>Year: {paper['year']}</span>
                            </div>
                            <div style='margin-top: 0.4rem; font-size: 0.9rem;'>
                                <strong>Authors:</strong> {paper['authors']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_score, col_button = st.columns([4, 1])
                        with col_score:
                            st.markdown(f"<span class='score-label'>Match Score: {paper['similarity_score']*100:.1f}%</span>", unsafe_allow_html=True)
                            st.progress(paper['similarity_score'])
                            
                        with col_button:
                            all_fav_ids = [f['paper_id'] for f in fetch_favorites()]
                            if paper['paper_id'] in all_fav_ids:
                                st.button("Saved ✓", key=f"fav_btn_pdf_s_{paper['paper_id']}", disabled=True)
                            else:
                                if st.button("Bookmark", key=f"fav_btn_pdf_{paper['paper_id']}"):
                                    add_to_favorites(paper)
                                    st.success("Saved!")
                                    st.rerun()
                                    
                        with st.expander("View Abstract"):
                            st.write(paper['abstract'])
                            if 'keywords' in paper and paper['keywords']:
                                st.write("**Paper Keywords:**")
                                tags = "".join([f"<span class='keyword-tag'>{kw}</span>" for kw in paper['keywords']])
                                st.markdown(tags, unsafe_allow_html=True)
                        st.markdown("---")
            else:
                st.info("No items mapped onto current threshold constraints.")

# TAB 3: BOOKMARKS & SAVED PAPERS
with tab_favs:
    st.subheader("Saved Reference Collection")
    fav_list = fetch_favorites()
    
    if fav_list:
        fav_df = pd.DataFrame(fav_list)
        if 'id' in fav_df.columns:
            fav_df = fav_df.drop(columns=['id'])
            
        st.dataframe(fav_df, use_container_width=True)
        
        for f_paper in fav_list:
            col_info, col_action = st.columns([5, 1])
            with col_info:
                st.markdown(f"**{f_paper['title']}** ({f_paper['year']})  \n`Ref ID: {f_paper['paper_id']}`")
            with col_action:
                if st.button("Delete", key=f"rem_fav_{f_paper['paper_id']}"):
                    remove_from_favorites(f_paper['paper_id'])
                    st.success("Deleted from list")
                    st.rerun()
            st.markdown("---")
    else:
        st.info("No items have been saved to your workspace library yet.")
