import asyncio
import traceback

from beeai_framework.agents.bee.agent import BeeAgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import SystemMessage, UserMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.workflows.agent import AgentFactoryInput, AgentWorkflow

async def main() -> None:
    try:
        llm = await ChatModel.from_name("ollama:granite3.1-dense:8b")
        
        workflow = AgentWorkflow(name="Recommendation System")
        
        workflow.add_agent(
            AgentFactoryInput(
                name="ResearchAgent",
                instructions="""You are a research specialist. Search for detailed information. For each find its:
                - Full name and location
                - Specialties and unique features
                - Customer reviews and expert opinions""",
                tools=[DuckDuckGoSearchTool(max_results=3)],
                llm=llm,
                memory=UnconstrainedMemory(),
                execution=BeeAgentExecutionConfig(
                    max_iterations=5,
                    max_retries_per_step=2,
                    total_max_retries=10
                )
            )
        )
        
        workflow.add_agent(
            AgentFactoryInput(
                name="WritingAgent",
                instructions="""You are a recommendation writer. Create EXACTLY 3 recommendations using this format:
                ### [Business Name]
                Description: [2-3 sentences about what makes this place special]
                Why Visit: [1-2 specific, compelling reasons to visit]""",
                llm=llm,
                memory=UnconstrainedMemory(),
                execution=BeeAgentExecutionConfig(
                    max_iterations=5,
                    max_retries_per_step=2,
                    total_max_retries=10
                )
            )
        )
        
        workflow.add_agent(
            AgentFactoryInput(
                name="ReviewAgent",
                instructions="""You are a formatting specialist. 

                Your output must exactly match this format:
                
                ### [Title]
                Description: [Description text]
                Why Visit: [Reasons text]""",
                llm=llm,
                memory=UnconstrainedMemory(),
                execution=BeeAgentExecutionConfig(
                    max_iterations=5,
                    max_retries_per_step=2,
                    total_max_retries=10
                )
            )
        )
        
        user_query = input("What would you like recommendations for? (e.g., 'best cafes in NYC'): ")
        
        memory = UnconstrainedMemory()
        await memory.add(SystemMessage("You are a specialized recommendation system."))
        await memory.add(UserMessage(content=user_query))
        
        result = await workflow.run(messages=memory.messages)
        
        print("\n" + "="*50 + "\n")
        print("RECOMMENDATIONS:\n")
        print(result.state.final_answer)
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
