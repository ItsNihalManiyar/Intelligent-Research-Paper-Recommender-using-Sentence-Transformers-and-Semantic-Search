import os
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
import time
import ssl
from typing import List, Dict, Any

# Bypass SSL certificate validation for macOS environments
try:
    # pyrefly: ignore[bad-assignment]
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

def fetch_arxiv_batch(start: int, max_results: int) -> List[Dict[str, Any]]:
    """
    Fetches a batch of papers from the arXiv API.
    """
    # Query for machine learning, deep learning, computer vision, and NLP papers
    query = "cat:cs.LG+OR+cat:cs.CV+OR+cat:cs.CL+OR+cat:cs.AI+OR+cat:stat.ML"
    url = f"http://export.arxiv.org/api/query?search_query={query}&start={start}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    
    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read()
    except Exception as e:
        print(f"Error fetching batch starting at {start}: {e}")
        return []
        
    root = ET.fromstring(xml_data)
    
    # Namespaces used in arXiv Atom feed
    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }
    
    papers = []
    for entry in root.findall('atom:entry', ns):
        # Extract ID (e.g., http://arxiv.org/abs/1909.00001v1 -> 1909.00001v1)
        raw_id_elem = entry.find('atom:id', ns)
        raw_id = raw_id_elem.text if raw_id_elem is not None and raw_id_elem.text else ""
        paper_id = raw_id.split('/abs/')[-1] if '/abs/' in raw_id else raw_id
        
        # Extract Title (clean up whitespace/newlines)
        title_elem = entry.find('atom:title', ns)
        title = " ".join(title_elem.text.split()) if title_elem is not None and title_elem.text else ""
        
        # Extract Abstract/Summary
        summary_elem = entry.find('atom:summary', ns)
        abstract = " ".join(summary_elem.text.split()) if summary_elem is not None and summary_elem.text else ""
        
        # Extract Authors
        authors = []
        for author in entry.findall('atom:author', ns):
            name_elem = author.find('atom:name', ns)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text)
        authors_str = ", ".join(authors)
        
        # Extract Published Year
        published_elem = entry.find('atom:published', ns)
        year = int(published_elem.text[:4]) if published_elem is not None and published_elem.text else None
        
        # Extract Category (Primary category)
        primary_cat_elem = entry.find('arxiv:primary_category', ns)
        if primary_cat_elem is not None:
            category = primary_cat_elem.attrib.get('term', 'unknown')
        else:
            # Fallback to general category term
            cat_elem = entry.find('atom:category', ns)
            category = cat_elem.attrib.get('term', 'unknown') if cat_elem is not None else 'unknown'
            
        papers.append({
            'paper_id': paper_id,
            'title': title,
            'abstract': abstract,
            'authors': authors_str,
            'year': year,
            'category': category
        })
        
    return papers

def download_dataset(total_papers: int = 5000, batch_size: int = 1000) -> None:
    """
    Downloads paper metadata from arXiv and saves it to data/raw/arxiv_raw.csv.
    """
    print(f"Starting download of {total_papers} papers from arXiv...")
    all_papers = []
    
    for start in range(0, total_papers, batch_size):
        current_batch_size = min(batch_size, total_papers - start)
        print(f"Fetching papers {start} to {start + current_batch_size}...")
        
        # Retry logic for robust downloading
        retries = 3
        batch = []
        for attempt in range(retries):
            batch = fetch_arxiv_batch(start, current_batch_size)
            if batch:
                break
            print(f"Attempt {attempt + 1} failed. Retrying in 5 seconds...")
            time.sleep(5)
            
        all_papers.extend(batch)
        print(f"Fetched {len(batch)} papers. Total so far: {len(all_papers)}")
        
        # Respect arXiv API rate limit policies (wait 3 seconds between requests)
        time.sleep(3)
        
    # Convert to DataFrame
    df = pd.DataFrame(all_papers)
    
    # Create output directory
    os.makedirs("data/raw", exist_ok=True)
    raw_path = "data/raw/arxiv_raw.csv"
    
    # Save raw dataset
    df.to_csv(raw_path, index=False)
    print(f"Successfully downloaded {len(df)} papers and saved to {raw_path}")

if __name__ == "__main__":
    download_dataset(total_papers=5000, batch_size=1000)
