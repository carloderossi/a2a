"""
LangChain Planner Agent with A2A Protocol

This agent uses LangChain with Ollama to create structured plans based on
research results and integrates with the A2A protocol.

Features:
- LangChain-based planning architecture
- Multiple Ollama model support
- Structured output generation
- A2A protocol compliance
- MCP agent card registration

Run:
    python agents/langchain_planner_agent.py
"""

import sys
from pathlib import Path
import asyncio
import atexit
import uuid
import httpx
import uvicorn
import logging
from typing import List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field
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
    LANGCHAIN_PLANNER_AGENT,
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

class ActionPlan(BaseModel):
    """Structured output schema for action plans."""
    objective: str = Field(description="The main objective of the plan")
    steps: List[str] = Field(description="List of actionable steps")
    timeline: str = Field(description="Estimated timeline for completion")
    resources_needed: List[str] = Field(description="Resources required")
    success_criteria: List[str] = Field(description="Criteria to measure success")


class LangChainPlannerExecutor:
    """
    Planner agent executor using LangChain and Ollama.
    
    This executor transforms research results into structured, actionable plans
    using LangChain's ChatOllama with optional structured outputs.
    """
    
    def __init__(
        self,
        model_name: str = OLLAMA_SLM,
        temperature: float = 0.3,
        use_structured_output: bool = False
    ):
        """
        Initialize the LangChain planner executor.
        
        Args:
            model_name: Ollama model to use (default: llama3.1)
            temperature: Sampling temperature (default: 0.3 for more focused planning)
            use_structured_output: Whether to use structured output (requires compatible model)
        """
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
        )
        self.use_structured_output = use_structured_output
        
        if use_structured_output:
            self.structured_llm = self.llm.with_structured_output(ActionPlan)
            logging.info(f"Initialized LangChain with structured output: {model_name}")
        else:
            logging.info(f"Initialized LangChain planner with model: {model_name}")
    
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """
        Execute planning based on research input.
        
        This method:
        1. Extracts research results from the request context
        2. Creates a planning prompt
        3. Generates a structured plan using LangChain
        4. Returns the plan via the A2A event queue
        
        Args:
            context: A2A request context with research input
            event_queue: A2A event queue for publishing responses
        """
        # Get the research input
        research = context.get_user_input()
        logging.info(f"Creating plan based on research: {research[:100]}...")
        
        # Create planning prompt
        prompt = f"""Based on the following research findings:

{research}

Create a comprehensive, actionable plan that includes:
1. Clear objectives
2. Step-by-step actions
3. Timeline estimates
4. Required resources
5. Success criteria

Format the plan clearly with numbered steps and sections."""
        
        # Create messages
        messages = [
            SystemMessage(content="""You are an expert strategic planner. Your task is to:
            1. Analyze the research findings thoroughly
            2. Create a clear, actionable plan
            3. Break down complex goals into specific steps
            4. Consider resources and timelines realistically
            5. Define measurable success criteria
            
            Be practical, specific, and comprehensive."""),
            HumanMessage(content=prompt)
        ]
        
        # Generate plan
        logging.info(f"Generating plan with LangChain (Ollama: {self.llm.model})...")
        
        if self.use_structured_output:
            try:
                # Try structured output
                response = await self.structured_llm.ainvoke(prompt)
                output = self._format_structured_plan(response)
                logging.info("Generated structured plan successfully")
            except Exception as e:
                logging.warning(f"Structured output failed, using regular output: {e}")
                response = await self.llm.ainvoke(messages)
                output = response.content
        else:
            # Regular text output
            response = await self.llm.ainvoke(messages)
            output = response.content
        
        logging.info(f"Plan generated. Length: {len(output)} chars")
        
        # Create A2A message
        msg = Message(
            messageId=str(uuid.uuid4()),
            role="agent",
            parts=[Part(kind="text", text=output)],
            final=True,
        )
        
        # Enqueue response
        await event_queue.enqueue_event(msg)
        logging.info("Plan enqueued successfully")
    
    def _format_structured_plan(self, plan: ActionPlan) -> str:
        """
        Format a structured ActionPlan into readable text.
        
        Args:
            plan: The structured plan object
            
        Returns:
            Formatted plan as text
        """
        output = f"""# Action Plan

## Objective
{plan.objective}

## Timeline
{plan.timeline}

## Steps
"""
        for i, step in enumerate(plan.steps, 1):
            output += f"{i}. {step}\n"
        
        output += "\n## Required Resources\n"
        for resource in plan.resources_needed:
            output += f"- {resource}\n"
        
        output += "\n## Success Criteria\n"
        for criterion in plan.success_criteria:
            output += f"- {criterion}\n"
        
        return output
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """Handle task cancellation (not implemented for synchronous LangChain calls)."""
        pass


def create_agent_card() -> AgentCard:
    """
    Create the Agent Card for the LangChain Planner Agent.
    
    Returns:
        AgentCard: Configured agent card with skills and capabilities
    """
    skill = AgentSkill(
        id="langchain_planning",
        name="LangChain Planning",
        description="Create structured action plans using LangChain and Ollama",
        tags=["planning", "strategy", "langchain", "ollama"],
        examples=[
            "Create an implementation plan for a microservices migration",
            "Develop a research roadmap for AI safety",
            "Plan a product launch strategy"
        ],
    )
    
    agent_card = AgentCard(
        name=LANGCHAIN_PLANNER_AGENT,
        description="LangChain-powered planning agent using Ollama models",
        url="http://localhost:9002/",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[skill],
        version="1.0.0",
        capabilities=AgentCapabilities(),
    )
    
    return agent_card


def main():
    """
    Initialize and start the LangChain Planner Agent server.
    
    This function:
    1. Creates the agent card
    2. Registers it with the MCP server
    3. Initializes the A2A server with LangChain executor
    4. Starts the server on port 9002
    """
    print(f"{GREEN}{'='*60}")
    print("LangChain Planner Agent")
    print(f"{'='*60}{RESET}\n")
    
    # Create agent card
    agent_card = create_agent_card()
    
    # Create the LangChain-based executor
    # You can enable structured output with compatible models
    executor = LangChainPlannerExecutor(
        model_name=OLLAMA_SLM,
        temperature=0.3,  # Lower temperature for more focused planning
        use_structured_output=False  # Set to True if model supports it
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
        asyncio.run(register_agentcard(agent_card, agent_id="langchain_planner_agent"))
        print(f"{GREEN}✓ Agent card registered{RESET}")
    except httpx.ConnectError as e:
        print(f"{YELLOW}⚠ Failed to connect to MCP Server: {e}{RESET}")
        print(f"{YELLOW}  Agent will run without MCP registration{RESET}")
    
    # Register cleanup on exit
    atexit.register(
        lambda: asyncio.run(remove_agentcard(agent_name=LANGCHAIN_PLANNER_AGENT))
    )
    
    # Start server
    print(f"\n{GREEN}Starting LangChain Planner Agent on port 9002...{RESET}")
    print(f"{CYAN}Agent Card available at: http://localhost:9002/.well-known/agent-card.json{RESET}\n")
    
    uvicorn.run(server.build(), host="0.0.0.0", port=9002)


if __name__ == "__main__":
    main()
