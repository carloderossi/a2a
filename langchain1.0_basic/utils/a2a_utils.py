"""
A2A Utilities for LangChain Integration

This module provides utilities for integrating A2A protocol with LangChain agents.
It includes functions for agent card management, resolution, and client creation.
"""

from fastmcp import Client
from mcp.types import TextContent
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.client import ClientFactory, ClientConfig, A2AClient
import json
import httpx

# MCP Server configuration
MCP_SERVER_URL = "http://127.0.0.1:8080/mcp"

# Agent names for the LangChain examples
LANGCHAIN_RESEARCH_AGENT = "cdr.LangChain_Research_Agent"
LANGCHAIN_PLANNER_AGENT = "cdr.LangChain_Planner_Agent"


async def register_agentcard(agentcard: AgentCard, agent_id: str):
    """
    Register an agent card with the MCP server for discovery.

    Args:
        agentcard (AgentCard): The agent card to register
        agent_id (str): Unique identifier for the agent

    Example:
        >>> card = AgentCard(name="my-agent", description="My agent", ...)
        >>> await register_agentcard(card, "my_agent_id")
    """
    client = Client(MCP_SERVER_URL)
    json_card = agentcard.model_dump()
    print(f"Registering agent card for agent ID: {agent_id}")
    print(f"Card: {json.dumps(json_card, indent=2)}")
    
    async with client:
        result = await client.call_tool("register_agent", {"card": json_card})
        print("Register result:", result)
        
        if result and isinstance(result[0], TextContent):
            message = result[0].text
            print("Register message:", message)

            if "Registered" in message:
                print(f"✓ Agent card registered successfully for agent ID: {agent_id}")
            else:
                print(f"⚠ Unexpected response: {message}")
        else:
            print(f"⚠ Unexpected result type: {result}")


async def remove_agentcard(agent_name: str):
    """
    Remove an agent card from the MCP server.

    Args:
        agent_name (str): Name of the agent to deregister

    Example:
        >>> await remove_agentcard("cdr.My_Agent")
    """
    print(f"Removing agent card for agent name: {agent_name}")
    client = Client(MCP_SERVER_URL)
    
    async with client:
        result = await client.call_tool("deregister_agent", {"name": agent_name})
        print("Deregister result:", result)
        
        if result and isinstance(result[0], TextContent):
            message = result[0].text
            print("Deregister message:", message)

            if "Deregistered" in message:
                print(f"✓ Agent card deregistered successfully: {agent_name}")
            else:
                print(f"⚠ Unexpected response: {message}")
        else:
            print(f"⚠ Unexpected result type: {result}")


async def resolve_agent_card(agent_uri: str, debug: bool = False) -> AgentCard:
    """
    Retrieve an Agent Card as a Resource from the MCP server.

    Agent Cards are standardized metadata documents that describe an agent's
    capabilities, skills, supported modes, and version information following
    the A2A protocol specification.

    Args:
        agent_uri (str): The URI of the agent card resource in the MCP Server
                        Format: "agent://agent-name"
        debug (bool): If True, print debug information

    Returns:
        AgentCard: The retrieved and validated Agent Card object

    Raises:
        ValueError: If the response cannot be parsed or is invalid

    Example:
        >>> card = await resolve_agent_card("agent://cdr.Research_Agent")
        >>> print(card.name, card.description)
    """
    async with Client(MCP_SERVER_URL) as client:
        response = await client.read_resource(agent_uri)
        
        # Handle different response types
        contents = response if isinstance(response, list) else [response]
        
        for content in contents:
            if hasattr(content, "text") and content.text:
                try:
                    card_data = json.loads(content.text)
                    
                    if debug:
                        print(f"Debug - Raw card data: {json.dumps(card_data, indent=2)}")
                    
                    # Validate and create AgentCard
                    card = AgentCard.model_validate(card_data)
                    
                    if debug:
                        print(f"Debug - Card attributes: {dir(card)}")
                    
                    return card
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse Agent Card JSON: {e}")
                except Exception as e:
                    if debug:
                        print(f"Debug - Validation error: {e}")
                        print(f"Debug - Card data keys: {card_data.keys()}")
                    raise ValueError(f"Failed to validate Agent Card: {e}")
        
        raise ValueError("No valid content found for Agent Card.")


def get_a2a_client(
    card: AgentCard,
    request_timeout: float = 360.0,
    connect_timeout: float = 60.0
) -> tuple[A2AClient, httpx.AsyncClient]:
    """
    Create and configure an A2A client for communicating with an agent.

    The ClientFactory automatically selects the appropriate client type
    (HTTP, gRPC, etc.) based on the agent card's transport protocol.

    Args:
        card (AgentCard): Agent card with connection information
        request_timeout (float): Timeout for individual requests (default: 360s)
        connect_timeout (float): Timeout for establishing connections (default: 60s)

    Returns:
        tuple[A2AClient, httpx.AsyncClient]: 
            - Configured A2A client for sending messages
            - Underlying HTTP client (must be closed with aclose())

    Example:
        >>> card = await resolve_agent_card("agent://my-agent")
        >>> client, httpx_client = get_a2a_client(card)
        >>> try:
        ...     response = await client.send_message(message)
        ... finally:
        ...     await httpx_client.aclose()
    """
    # Create httpx client with custom timeout
    httpx_client = httpx.AsyncClient(
        timeout=httpx.Timeout(request_timeout, connect=connect_timeout)
    )
    
    # Configure client factory
    config = ClientConfig(
        streaming=False,  # Preference, but may still stream if agent requires it
        httpx_client=httpx_client
    )
    
    factory = ClientFactory(config)
    client = factory.create(card)
    
    print(f"✓ A2A Client initialized for agent: {card.name}")
    return client, httpx_client


def extract_text_from_response(response) -> str | None:
    """
    Extract text content from an A2A response object.

    Args:
        response: A2A response object with message parts

    Returns:
        str | None: Extracted text content or None if not found

    Example:
        >>> response = await client.send_message(msg)
        >>> text = extract_text_from_response(response)
        >>> print(text)
    """
    if not response:
        return None
        
    data = response.model_dump()
    
    try:
        kind = data.get("kind")
        if kind != "message":
            print(f"⚠ Unexpected result kind: {kind}")
            return None
            
        parts = data.get("parts", [])
        if not parts:
            return None
            
        first_part = parts[0]
        if first_part.get("kind") == "text":
            return first_part.get("text")
            
        return None
    except (KeyError, IndexError, TypeError) as e:
        print(f"⚠ Error extracting text: {e}")
        return None
