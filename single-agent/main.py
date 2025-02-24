import asyncio
import traceback
from pydantic import ValidationError

from beeai_framework.workflows.workflow import WorkflowError
from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from state import TravelState
from workflow import create_travel_workflow

async def main():
    try:
        llm = await ChatModel.from_name("ollama:granite3.1-dense:8b")
        search_tool = DuckDuckGoSearchTool(max_results=3)
        
        workflow = await create_travel_workflow(llm, search_tool)
        
        user_query = input("What would you like recommendations for? (e.g., 'tech conferences in Europe during spring 2025', 'warm winter destinations with direct flights from Boston', 'top beaches in Hawaii for snorkeling'): ")
        
        result = await workflow.run(TravelState(query=user_query))
        
        print("\n" + "="*50 + "\n")
        print("Search Strategy:")
        print(result.state.search_plan)
        print("\n" + "="*50 + "\n")
        print("Recommendations:")
        print(result.state.recommendations)
        
    except WorkflowError:
        traceback.print_exc()
    except ValidationError as e:
        print(f"Validation error: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
