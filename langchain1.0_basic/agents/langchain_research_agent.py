"""
LangChain Research Agent with A2A Protocol

This agent uses LangChain with Ollama to perform research tasks and integrates
with the A2A protocol for agent-to-agent communication.

Features:
- LangChain-based agent architecture
- Ollama LLM integration
- A2A protocol compliance
- MCP agent card registration

Run:
    python agents/langchain_research_agent.py
"""

import sys
from pathlib import Path
import asyncio
import atexit
import uuid
import httpx
import uvicorn
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Message, Part

from utils import (
    register_agentcard,
    remove_agentcard,
    LANGCHAIN_RESEARCH_AGENT,
)


# ANSI color codes
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

# Configure logging
logging.basicConfig(
    format=f'{CYAN}[%(filename)s - %(asctime)s]{RESET} {GREEN}%(levelname)s{RESET} %(message)s',
    datefmt='%m.%d.%Y %H:%M:%S',
    level=logging.INFO
)

OLLAMA_SLM = "qwen3" # Default Ollama model

class LangChainResearchExecutor:
    """
    Research agent executor using LangChain and Ollama.
    
    This executor processes research queries using LangChain's ChatOllama
    and returns structured responses following the A2A protocol.
    """
    
    def __init__(self, model_name: str = OLLAMA_SLM, temperature: float = 0.7):
        """
        Initialize the LangChain research executor.
        
        Args:
            model_name: Ollama model to use (default: llama3.1)
            temperature: Sampling temperature for generation (default: 0.7)
        """
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
        )
        logging.info(f"Initialized LangChain with model: {model_name}")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """
        Execute a research query using LangChain.
        
        This method:
        1. Extracts the query from the request context
        2. Processes it using LangChain with Ollama
        3. Returns the result via the A2A event queue
        
        Args:
            context: A2A request context with user input
            event_queue: A2A event queue for publishing responses
        """
        # Get the user's query
        query = context.get_user_input()
        logging.info(f"Processing research query: {query[:100]}...")
        
        # Create LangChain messages
        messages = [
            SystemMessage(content="""You are a research assistant. Your task is to:
            1. Analyze the query thoroughly
            2. Provide comprehensive, factual information
            3. Cite key concepts and findings
            4. Structure your response clearly
            
            Be concise but thorough."""),
            HumanMessage(content=query or "Provide general research guidance")
        ]
        
        # Get response from LangChain + Ollama
        logging.info(f"Sending query to LangChain (Ollama: {self.llm.model})...")
        response = await self.llm.ainvoke(messages)
        output = response.content
        
        logging.info(f"Research completed. Response length: {len(output)} chars")
        
        # Create A2A message
        msg = Message(
            messageId=str(uuid.uuid4()),
            role="agent",
            parts=[Part(kind="text", text=output)],
            final=True,
        )
        
        # Enqueue response
        await event_queue.enqueue_event(msg)
        logging.info("Response enqueued successfully")
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Handle task cancellation (not implemented for synchronous LangChain calls)."""
        pass


def create_agent_card() -> AgentCard:
    """
    Create the Agent Card for the LangChain Research Agent.
    
    Returns:
        AgentCard: Configured agent card with skills and capabilities
    """
    skill = AgentSkill(
        id="langchain_research",
        name="LangChain Research",
        description="Research and information gathering using LangChain and Ollama",
        tags=["research", "langchain", "ollama", "llm"],
        examples=[
            "Summarize recent advances in quantum computing",
            "Research best practices for microservices architecture",
            "Explain the principles of reinforcement learning"
        ],
    )
    
    agent_card = AgentCard(
        name=LANGCHAIN_RESEARCH_AGENT,
        description="LangChain-powered research agent using Ollama models",
        url="http://localhost:9001/",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[skill],
        version="1.0.0",
        capabilities=AgentCapabilities(),
    )
    
    return agent_card


def main():
    """
    Initialize and start the LangChain Research Agent server.
    
    This function:
    1. Creates the agent card
    2. Registers it with the MCP server
    3. Initializes the A2A server with LangChain executor
    4. Starts the server on port 9001
    """
    print(f"{GREEN}{'='*60}")
    print("LangChain Research Agent")
    print(f"{'='*60}{RESET}\n")
    
    # Create agent card
    agent_card = create_agent_card()
    
    # Create the LangChain-based executor
    executor = LangChainResearchExecutor(
        model_name=OLLAMA_SLM,
        temperature=0.7
    )
    
    # Create request handler
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    
    # Create A2A application
    server = A2AStarletteApplication(
        http_handler=request_handler,
        agent_card=agent_card,
    )
    
    # Register agent card with MCP
    try:
        logging.info("Registering agent card with MCP server...")
        asyncio.run(register_agentcard(agent_card, agent_id="langchain_research_agent"))
        print(f"{GREEN}✓ Agent card registered{RESET}")
    except httpx.ConnectError as e:
        print(f"{YELLOW}⚠ Failed to connect to MCP Server: {e}{RESET}")
        print(f"{YELLOW}  Agent will run without MCP registration{RESET}")
    
    # Register cleanup on exit
    atexit.register(
        lambda: asyncio.run(remove_agentcard(agent_name=LANGCHAIN_RESEARCH_AGENT))
    )
    
    # Start server
    print(f"\n{GREEN}Starting LangChain Research Agent on port 9001...{RESET}")
    print(f"{CYAN}Agent Card available at: http://localhost:9001/.well-known/agent-card.json{RESET}\n")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=9001)


if __name__ == "__main__":
    main()
