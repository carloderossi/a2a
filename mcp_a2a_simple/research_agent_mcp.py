"""
Research Agent Server

This module implements the Research Agent, an A2A-compliant server that processes
research queries and generates summaries using the Ollama LLM. It runs on port 9001
and follows the Agent-to-Agent (A2A) protocol specification.

A2A Protocol Reference:
    Official Documentation: https://a2a-protocol.org/
    GitHub Repository: https://github.com/google-a2a/A2A
    Agent Cards: https://a2a-protocol.org/latest/spec/#agent-cards
    Agent Skills: https://a2a-protocol.org/latest/spec/#agent-skills
"""

import httpx
import atexit
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from agent_executor import ResearchAgentExecutor

from mcp_utils import remove_agentcard, register_agentcard, REASEARCH_AGENT_NAME
import asyncio

def main():
    """
    Initialize and start the Research Agent server.
    
    This function:
    1. Defines the agent's research skill and capabilities per A2A spec
    2. Creates an Agent Card for service discovery and metadata
    3. Configures the request handler with the research executor
    4. Launches the server on port 9001
    
    A2A Concepts:
        - AgentSkill: Defines the agent's capabilities (what it can do)
        - AgentCard: Standardized metadata for discovery and interoperability
        - Request Handler: Manages A2A protocol lifecycle (JSON-RPC methods)
    
    A2A Reference:
        Server Implementation Guide: https://a2a-protocol.org/latest/guides/implementing-agents/
    """
    
    # Define the agent's primary skill
    # Skills are discoverable capabilities that describe what this agent can do
    # See: https://a2a-protocol.org/latest/spec/#agent-skills
    skill = AgentSkill(
        id="research",  # Unique identifier for this skill
        name="Research",  # Human-readable name
        description="Summarize or gather information using an LLM",  # Skill description
        tags=["research", "summarize", "ollama"],  # Tags for discovery and categorization
        examples=[  # Example queries to help users understand this skill's use cases
            "Summarize reinforcement learning", 
            "Find key points"
        ],
    )

    # Create the Agent Card - a standardized metadata document
    # This is served at /.well-known/agent-card.json for A2A service discovery
    # See: https://a2a-protocol.org/latest/spec/#agent-cards
    agent_card = AgentCard(
        name=REASEARCH_AGENT_NAME,  # Human-readable agent name
        description="Uses Ollama to summarize information",  # Agent's primary function
        url="http://localhost:9001/",  # The endpoint where this agent is accessible
        defaultInputModes=["text"],  # Supported input formats (text only)
        defaultOutputModes=["text"],  # Supported output formats (text only)
        skills=[skill],  # List of skills this agent provides
        version="1.0.0",  # Agent version for compatibility and change tracking
        capabilities=AgentCapabilities(),  # Additional capabilities (using framework defaults)
    )

    # Set up the request handler
    # This manages the lifecycle of incoming requests, task tracking, and responses
    # It implements the A2A JSON-RPC methods (message/send, tasks/get, etc.)
    request_handler = DefaultRequestHandler(
        agent_executor=ResearchAgentExecutor(),  # The executor containing business logic
        task_store=InMemoryTaskStore(),  # Storage for tracking asynchronous tasks
    )

    # Create the ASGI web application using Starlette
    # This provides the HTTP layer for the A2A protocol
    # All A2A communication happens via JSON-RPC 2.0 over HTTP(S)
    server = A2AStarletteApplication(
        http_handler=request_handler,  # Processes HTTP requests according to A2A protocol
        agent_card=agent_card,  # Agent metadata for discovery and introspection
    )
    
    # Register the Agent Card with the MCP server for discovery by other agents
    try:
        asyncio.run(register_agentcard(agent_card, agent_id="research_agent"))
    except httpx.ConnectError as e:
        print(f"Failed to connect to MCP Server: {e}")
        return

    # Ensure the agent card is removed from the MCP server on exit
    atexit.register(lambda: asyncio.run(remove_agentcard(agent_name=REASEARCH_AGENT_NAME)))

    # Launch the server
    # - host="0.0.0.0" makes it accessible from other machines (not just localhost)
    # - port=9001 is the designated port for the Research Agent
    # This enables remote A2A agents to communicate with this agent
    uvicorn.run(server.build(), host="0.0.0.0", port=9001)


if __name__ == "__main__":
    main()