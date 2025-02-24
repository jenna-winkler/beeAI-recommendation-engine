from beeai_framework.backend.message import UserMessage
from beeai_framework.workflows.workflow import Workflow
from state import TravelState, SearchInputSchema
from prompts import recommendations_template, get_planning_prompt
from utils import extract_search_queries

async def plan_step(state: TravelState, llm, **kwargs) -> str:
    """Plan search strategy for the query"""
    print(f"Planning approach for: {state.query}")
    
    planning_prompt = get_planning_prompt(state.query)
    plan_response = await llm.create({"messages": [UserMessage(content=planning_prompt)]})
    state.search_plan = plan_response.get_text_content()
    
    extracted_queries = extract_search_queries(state.search_plan)
    
    if extracted_queries:
        state.search_queries = extracted_queries
        print(f"Generated search queries: {state.search_queries}")
    else:
        state.search_queries = [state.query]
        print(f"Using default query: {state.query}")
    
    return "search_step"

async def search_step(state: TravelState, search_tool, **kwargs) -> str:
    """Execute searches based on the generated queries"""
    print(f"Executing search strategy...")
    
    all_results = []
    
    for query in state.search_queries:
        print(f"Searching for: {query}")
        search_output = search_tool.run({"query": query})
        
        query_results = [str(result) for result in search_output.results]
        all_results.append(f"Results for '{query}':\n" + "\n".join(query_results))
    
    state.search_results = "\n\n".join(all_results)
    
    return "recommend_step"

async def recommend_step(state: TravelState, llm, **kwargs) -> str:
    """Generate recommendations based on search results"""
    print("Generating recommendations...")
    
    prompt = recommendations_template.render(
        SearchInputSchema(query=state.query, search_results=state.search_results)
    )
    
    response = await llm.create({"messages": [UserMessage(content=prompt)]})
    state.recommendations = response.get_text_content()
    
    return Workflow.END

async def create_travel_workflow(llm, search_tool):
    """Create and configure the travel recommendation workflow"""
    workflow = Workflow(TravelState)
    
    async def bound_plan_step(state):
        return await plan_step(state, llm)
        
    async def bound_search_step(state):
        return await search_step(state, search_tool)
        
    async def bound_recommend_step(state):
        return await recommend_step(state, llm)
    
    workflow.add_step("plan_step", bound_plan_step)
    workflow.add_step("search_step", bound_search_step)
    workflow.add_step("recommend_step", bound_recommend_step)
    
    return workflow
