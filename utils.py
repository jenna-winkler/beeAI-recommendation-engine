import re

def extract_search_queries(text):
    """Extract search queries from planning text"""
    queries = re.findall(r'(?:Search Query:|search for:)\s*(?:"([^"]+)"|(.+)$)', 
                        text, re.MULTILINE | re.IGNORECASE)
    
    extracted_queries = [q[0] if q[0] else q[1] for q in queries if q[0] or q[1]]
    
    return extracted_queries