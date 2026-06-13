from keybert import KeyBERT
from typing import List

# Lazy initialization of KeyBERT model to avoid loading overhead when importing the module
_kw_model = None

def get_keyword_model() -> KeyBERT:
    """
    Returns the KeyBERT model, initializing it if it hasn't been already.
    """
    global _kw_model
    if _kw_model is None:
        # KeyBERT by default uses 'all-MiniLM-L6-v2' which is fast and lightweight
        _kw_model = KeyBERT()
    return _kw_model

def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """
    Extracts the top_n keywords/phrases from the input text using KeyBERT.
    """
    if not text or len(text.strip()) == 0:
        return []
        
    try:
        model = get_keyword_model()
        # Extract keywords with unigrams and bigrams
        keywords_with_scores = model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 2), 
            stop_words='english', 
            top_n=top_n
        )
        keywords: List[str] = []
        for item in keywords_with_scores:
            if isinstance(item, tuple) and len(item) >= 1:
                keywords.append(str(item[0]))
            elif isinstance(item, str):
                keywords.append(item)
        return keywords
    except Exception as e:
        print(f"KeyBERT extraction failed: {e}. Falling back to basic word splitting.")
        # Very simple fallback in case of out of memory or device issues
        words = [w.strip(".,;:?!'\"()").lower() for w in text.split()]
        stop_words = {'the', 'a', 'an', 'and', 'but', 'if', 'or', 'because', 'as', 'what', 'its', 'of', 'in', 'on', 'to', 'for', 'with', 'by', 'about', 'is', 'are', 'was', 'were', 'that', 'this', 'these', 'those', 'papers', 'paper', 'research', 'study', 'using', 'used', 'use', 'from'}
        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
        # Get unique words up to top_n
        seen = set()
        unique_words = []
        for w in filtered_words:
            if w not in seen:
                seen.add(w)
                unique_words.append(w)
            if len(unique_words) >= top_n:
                break
        return unique_words
