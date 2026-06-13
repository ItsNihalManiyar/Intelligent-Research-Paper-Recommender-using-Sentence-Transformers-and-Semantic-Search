import io
from typing import Union
import pypdf

def extract_text_from_pdf(pdf_source: Union[str, bytes, io.BytesIO], max_chars: int = 1500) -> str:
    """
    Extracts plain text from a PDF source (file path, bytes, or file-like object)
    and returns the first max_chars characters.
    """
    try:
        if isinstance(pdf_source, bytes):
            pdf_file = io.BytesIO(pdf_source)
        elif isinstance(pdf_source, str):
            pdf_file = open(pdf_source, 'rb')
        else:
            pdf_file = pdf_source
            
        reader = pypdf.PdfReader(pdf_file)
        text_content = []
        
        # Iterate over pages and extract text
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_content.append(text)
                
            # If we've already exceeded the character count, we can stop reading
            if sum(len(t) for t in text_content) >= max_chars:
                break
                
        # Close file if we opened it from a string path
        if isinstance(pdf_source, str):
            pdf_file.close()
            
        full_text = " ".join(text_content)
        # Normalize whitespace
        clean_text = " ".join(full_text.split())
        
        return clean_text[:max_chars]
        
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""
