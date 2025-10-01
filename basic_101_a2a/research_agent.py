# research_agent.py (port 9001)
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from agent_executor import ResearchAgentExecutor  # you implement this

def main():
    skill = AgentSkill(
        id="research",
        name="Research",
        description="Summarize or gather information using an LLM",
        tags=["research", "summarize", "ollama"],
        examples=["Summarize reinforcement learning", "Find key points"],
    )

    agent_card = AgentCard(
        name="Research Agent",
        description="Uses Ollama to summarize information",
        url="http://localhost:9001/",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[skill],
        version="1.0.0",
        capabilities=AgentCapabilities(),
    )

    request_handler = DefaultRequestHandler(
        agent_executor=ResearchAgentExecutor(),  # calls ollama.chat()
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        http_handler=request_handler,
        agent_card=agent_card,
    )

    uvicorn.run(server.build(), host="0.0.0.0", port=9001)

if __name__ == "__main__":
    main()