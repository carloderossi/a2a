"""
Client Agent Module

This module implements a client that orchestrates communication between multiple A2A agents.
It demonstrates a pipeline where a Research Agent processes a query, and a Planner Agent
creates an actionable plan based on the research results.

A2A Protocol Reference:
    Official Documentation: https://a2a-protocol.org/
    GitHub Repository: https://github.com/google-a2a/A2A
    JSON-RPC Methods: https://a2a-protocol.org/latest/spec/#json-rpc-methods
    Agent Cards: https://a2a-protocol.org/latest/spec/#agent-cards
"""

import requests
import uuid
import time
import logging
import httpx
from a2a.client.transports.jsonrpc import JsonRpcTransport
from a2a.client import A2ACardResolver, A2AClient, ClientFactory, ClientConfig, A2AGrpcClient
from a2a.types import (
    AgentCard,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    TextPart,
)

from mcp_utils import resolve_agent_card, REASEARCH_AGENT_NAME, PLANNER_AGENT_NAME

# URLs for the two agent servers
# RESEARCH_URL = "http://localhost:9001/"
# PLANNER_URL = "http://localhost:9002/"

# ANSI escape codes for colored terminal output
LIGHT_BLUE = '\033[94m'
CYAN = '\033[96m'
GREEN = '\033[92m'
RESET = '\033[0m'

# Configure logging with colored output for better visibility
logging.basicConfig(
    format=f'{CYAN}[%(filename)s - %(asctime)s]{RESET} {GREEN}%(levelname)s{RESET} %(message)s',
    datefmt='%m.%d.%Y %H:%M:%S',
    level=logging.INFO
)

def extract_text(resp) -> str | None:
    """Extracts the text content from an A2A response object."""
    data = resp.model_dump()
    try:
        kind = data.get("kind")
        if kind != "message":
            logging.error(f"Unexpected result kind: {kind}")
            return None
        parts = data.get("parts", [])
        if not parts:
            return None
        first_part = parts[0]
        if first_part.get("kind") == "text":
            return first_part.get("text")
        return None
    except (KeyError, IndexError, TypeError):
        return None

def get_client(card:AgentCard, request_timeout=360.0, connect_timeout=60.0) -> A2AClient:
    """
    Create and configure an A2A client for communicating with an agent.

    This function initializes an A2A client with custom timeout settings using the
    ClientFactory pattern. The factory automatically selects the appropriate client
    type based on the agent card's transport protocol (HTTP, gRPC, etc.).

    Args:
        card (AgentCard): The agent card containing metadata and connection information
            for the target agent. Typically obtained via resolve_agent_card().
        request_timeout (float, optional): Timeout in seconds for individual requests.
            Defaults to 360.0 (6 minutes).
        connect_timeout (float, optional): Timeout in seconds for establishing connections.
            Defaults to 60.0 (1 minute).

    Returns:
        tuple: A tuple containing:
            - A2AClient: Configured A2A client instance for sending messages
            - httpx.AsyncClient: The underlying HTTP client (must be closed with aclose())

    Example:
        >>> card = await resolve_agent_card("agent://research-agent")
        >>> client, httpx_client = get_client(card)
        >>> try:
        ...     async for chunk in client.send_message(message):
        ...         response = chunk
        ... finally:
        ...     await httpx_client.aclose()  # Always clean up

    Note:
        - The streaming preference is set to False, but agents may still stream
          if their implementation requires it
        - The returned httpx_client must be properly closed using aclose() to
          prevent resource leaks
        - Custom timeouts are useful for long-running agent operations

    A2A Reference:
        Client Configuration: https://github.com/google-a2a/A2A
    """
    # Create httpx client with custom timeout
    httpx_client = httpx.AsyncClient(
        timeout=httpx.Timeout(request_timeout, connect=connect_timeout)
    )
    # Configure client factory
    config = ClientConfig(
        streaming=False,  # Preference, but may still stream if agent requires it
        httpx_client=httpx_client  # Pass custom httpx client
    )
    factory = ClientFactory(config)
    client = factory.create(card)
    print("A2AClient initialized via ClientFactory")
    return client, httpx_client

async def run_messagev2(card: AgentCard, query):
    """
    Send a message to an agent and wait for the complete response.
    
    This function handles two possible response modes defined by the A2A protocol:
    1. Immediate response: Agent returns the result directly (synchronous)
    2. Async task: Agent returns a task ID, requiring polling for completion
    
    Args:
        base_url (str): Base URL of the agent server
        text (str): The message text to send to the agent
    
    Returns:
        str: The agent's response text
    
    Raises:
        RuntimeError: If the task fails or returns an unexpected result format
    
    A2A Reference:
        Message/Send method: https://a2a-protocol.org/latest/spec/#messagesend
        Task Lifecycle: https://a2a-protocol.org/latest/spec/#task-lifecycle
        Task states: submitted → working → succeeded/failed
    """    
    client, httpx_client = get_client(card)
    print("Creating message payload...")
    #httpx_client = client.httpx_client  # Access the underlying httpx client
    try:
        message_payload = Message(
            role=Role.user,
            messageId=str(uuid.uuid4()),
            parts=[Part(root=TextPart(text=query))],
        )
        print(f"Message payload constructed: \n{message_payload}")
        
        # Don't wrap in SendMessageRequest - pass Message directly
        print(f"Sending message to {card.name}...")
        
        async def collect_response():
            response = None
            async for chunk in client.send_message(message_payload):
                response = chunk
                print(f"Received chunk")
            return response
        
        response = await asyncio.wait_for(collect_response(), timeout=360)
        
        if response:
            print("Response:")
            print(response.model_dump_json(indent=2))
            return extract_text(response)
        else:
            print("No response received")
            return None
    finally:
        await httpx_client.aclose()  # Clean up the clientponse received")

async def fetch_agent_card(agent_uri: str) -> AgentCard:
    """
    Retrieve the Agent Card from an A2A-compliant agent server.
    
    Agent Cards are standardized metadata documents that describe an agent's
    capabilities, skills, supported modes, and version information. They follow
    the A2A protocol specification and are served at a well-known endpoint.
    
    Args:
        base_url (str): Base URL of the agent server (e.g., "http://localhost:9001/")
    
    Returns:
        dict: The parsed Agent Card JSON containing agent metadata
    
    Raises:
        requests.HTTPError: If the request fails or returns an error status
    
    A2A Reference:
        Agent Card Specification: https://a2a-protocol.org/latest/spec/#agent-cards
        Well-known endpoint format: {base_url}/.well-known/agent-card.json
    """
    async with httpx.AsyncClient() as httpx_client:
        logging.info(f"Fetching Agent Card from {agent_uri}...")
        card = await resolve_agent_card(agent_uri)
        return card

async def main():
    """
    Main execution function demonstrating A2A agent orchestration.
    
    This function demonstrates a multi-agent workflow using the A2A protocol:
    1. Fetches and displays Agent Cards from both agents (service discovery)
    2. Sends a research query to the Research Agent
    3. Forwards the research results to the Planner Agent
    4. Displays the final plan
    
    This creates a simple two-stage pipeline: Research → Planning
    
    A2A Concepts Demonstrated:
        - Agent discovery via Agent Cards
        - Synchronous message passing between agents
        - Agent chaining/pipelining for complex workflows
    
    A2A Reference:
        Multi-agent workflows: https://a2a-protocol.org/latest/guides/
    """
    # Fetch metadata from both agents to verify they're available and properly configured
    # This demonstrates the A2A service discovery pattern using Agent Cards
    try:
        research_card = await resolve_agent_card("agent://" + REASEARCH_AGENT_NAME)
        planner_card = await resolve_agent_card("agent://" + PLANNER_AGENT_NAME)
    except httpx.ConnectError as e:
        logging.error(f"Failed to connect to MCP Server: {e}")
        return    
    logging.info(f"Research AgentCard: {research_card.name}")
    print(research_card.model_dump_json(indent=2))
    logging.info(f"Planner AgentCard: {planner_card.name}")
    print(planner_card.model_dump_json(indent=2))

    # Define the initial user query
    query = "Summarize the latest approaches to reinforcement learning exploration."
    logging.info(f"\n[User Query]\n {query}")
    
    # Stage 1: Send query to Research Agent for information gathering
    logging.info("Sending query to Research agent...")
    research_result = await run_messagev2(research_card, query)
    logging.info(f"\n[Research Result]\n{research_result}\n")

    # Stage 2: Send research results to Planner Agent for plan generation
    # This demonstrates agent chaining - output from one agent becomes input to another
    logging.info("Sending research result to Planner agent...")
    plan_result = await run_messagev2(planner_card, research_result)
    logging.info(f"\n[Planner Result]\n{plan_result}\n")

    logging.info("Done.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


"""
Commented out code from the original file:
This section was likely used for debugging or an alternative logging approach.

logging.info("Research AgentCard:", research_card["name"])
logging.info(research_card)
logging.info("Planner AgentCard:", planner_card["name"])
logging.info(planner_card)
"""