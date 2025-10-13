"""
Planner Agent Server

This module implements the Planner Agent, an A2A-compliant server that transforms
research results into structured, actionable plans using the Ollama LLM. It runs
on port 9002 and follows the Agent-to-Agent (A2A) protocol specification.

A2A Protocol Reference:
    Official Documentation: https://a2a-protocol.org/
    GitHub Repository: https://github.com/google-a2a/A2A
    Agent Cards: https://a2a-protocol.org/latest/spec/#agent-cards
    Agent Skills: https://a2a-protocol.org/latest/spec/#agent-skills
"""

import httpx
import uvicorn
import atexit
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from agent_executor import PlannerAgentExecutor

from mcp_utils import remove_agentcard, register_agentcard, PLANNER_AGENT_NAME
import asyncio

def main():
    """
    Initialize and start the Planner Agent server.
    
    This function:
    1. Defines the agent's skill set and capabilities per A2A specification
    2. Creates an Agent Card for service discovery
    3. Configures the request handler with the executor
    4. Launches the server on port 9002
    
    A2A Concepts:
        - AgentSkill: Defines what the agent can do (capabilities)
        - AgentCard: Machine-readable metadata for discovery and interoperability
        - Request Handler: Manages the A2A protocol request/response lifecycle
    
    A2A Reference:
        Server Implementation: https://a2a-protocol.org/latest/guides/implementing-agents/
    """
    
    # Define the agent's primary skill
    # Skills describe what the agent can do and help other agents discover its capabilities
    # See: https://a2a-protocol.org/latest/spec/#agent-skills
    skill = AgentSkill(
        id="planning",  # Unique identifier for this skill
        name="Planning",  # Human-readable name
        description="Turn research into a step-by-step plan",  # What this skill does
        tags=["planning", "strategy", "ollama"],  # Searchable tags for discovery
        examples=[  # Example use cases to help users/agents understand when to use this skill
            "Make an experiment plan", 
            "Outline next steps"
        ],
    )

    # Create the Agent Card - a standardized metadata document
    # This card is served at /.well-known/agent-card.json for service discovery
    # See: https://a2a-protocol.org/latest/spec/#agent-cards
    agent_card = AgentCard(
        name=PLANNER_AGENT_NAME,  # Human-readable agent name
        description="Uses Ollama to create structured plans",  # Agent's purpose
        url="http://localhost:9002/",  # Where this agent can be reached
        defaultInputModes=["text"],  # Input formats this agent accepts (text only)
        defaultOutputModes=["text"],  # Output formats this agent produces (text only)
        skills=[skill],  # List of skills this agent provides
        version="1.0.0",  # Agent version for compatibility tracking
        capabilities=AgentCapabilities(),  # Additional capabilities (using defaults)
    )

    # Configure the request handler
    # This component manages the A2A protocol request/response lifecycle and task execution
    # It handles JSON-RPC method calls and routes them to the appropriate executor
    request_handler = DefaultRequestHandler(
        agent_executor=PlannerAgentExecutor(),  # The executor that does the actual work
        task_store=InMemoryTaskStore(),  # In-memory storage for tracking async tasks
    )

    # Create the Starlette-based ASGI application
    # This wraps the A2A protocol handling in a web server
    # Starlette provides the HTTP layer for the A2A JSON-RPC communication
    server = A2AStarletteApplication(
        http_handler=request_handler,  # Handles incoming HTTP requests
        agent_card=agent_card,  # Agent metadata for discovery
    )

    # Register the Agent Card with the MCP server for discovery by other agents
    try:
        asyncio.run(register_agentcard(agent_card, agent_id="planner_agent"))
    except httpx.ConnectError as e:
        print(f"Failed to connect to MCP Server: {e}")
        return
    # Ensure the agent card is removed from the MCP server on exit
    atexit.register(lambda: asyncio.run(remove_agentcard(agent_name=PLANNER_AGENT_NAME)))

    # Start the server on port 9002
    # host="0.0.0.0" makes it accessible from other machines on the network
    # This allows remote agents to communicate with this agent over HTTP
    uvicorn.run(server.build(), host="0.0.0.0", port=9002)

if __name__ == "__main__":
    main()