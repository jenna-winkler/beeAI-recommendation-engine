from pydantic import BaseModel

class TravelState(BaseModel):
    query: str
    search_plan: str = ""
    search_queries: list[str] = []
    search_results: str = ""
    recommendations: str = ""

class SearchInputSchema(BaseModel):
    query: str
    search_results: str
