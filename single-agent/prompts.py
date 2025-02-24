from beeai_framework.utils.templates import PromptTemplate
from state import SearchInputSchema

recommendations_template = PromptTemplate(
    schema=SearchInputSchema,
    template="""
    Based on the following search results, provide the top 3-5 recommendations for: "{{query}}"
    
    Search Results:
    {{search_results}}
    
    For each recommendation, include:
    - Name
    - Brief description
    - Why it's worth visiting
    
    Format each recommendation with a clear heading and keep the overall response concise.
    """
)

def get_planning_prompt(query):
    return f"""I need to provide helpful recommendations about '{query}'. 
    What specific information should I search for to give the best recommendations?
    
    Generate a search strategy with 1-3 specific search queries that will help me find comprehensive information.
    Format each search query as "Search Query: [query]" on a new line.
    """
