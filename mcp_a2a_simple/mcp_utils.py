from fastmcp import Client
from mcp.types import TextContent
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
import json

MCP_SERVER_URL="http://127.0.0.1:8080/mcp"
PLANNER_AGENT_NAME="cdr.Planner_Agent"
REASEARCH_AGENT_NAME="cdr.Research_Agent"

async def register_agentcard(agentcard, agent_id):
    """Register an agent card with a specific agent ID.

    Args:
        agentcard: The agent card to register.
        agent_id: The ID of the agent to associate with the card.
    """
    client = Client(MCP_SERVER_URL)
    json_card = agentcard.model_dump()
    print(f"Registering agent card for agent ID: {agent_id} with card:\n{json_card}")
    async with client:
        result = await client.call_tool("register_agent", {"card": json_card})
        print("Register result:", result)
        if result and isinstance(result[0], TextContent):
            message = result[0].text
            print("Register message:", message)

            if "Registered" in message:
                print(f"Agent card registered successfully for agent ID: {agent_id}")
            else:
                print(f"Unexpected response while registering agent {agent_id}: {message}")
        else:
            print(f"Unexpected result type: {result}")

async def remove_agentcard(agent_name):
    """Remove an agent card associated with a specific agent name.

    Args:
        agent_name: The Name of the agent whose card should be removed.
    """
    print(f"Removing agent card for agent Name: {agent_name}")
    client = Client(MCP_SERVER_URL)
    async with client:
        result = await client.call_tool("deregister_agent", {"name": agent_name})
        print("Deregister result:", result)
        if result and isinstance(result[0], TextContent):
            message = result[0].text
            print("Deregister message:", message)

            if "Deregistered" in message:
                print(f"Agent card deregistered successfully for agent Name: {agent_name}")
            else:
                print(f"Unexpected response while deregistering agent {agent_name}: {message}")
        else:
            print(f"Unexpected result type: {result}")

async def resolve_agent_card(agent_uri) -> AgentCard:
    """
    Retrieve the Agent Card as a Resource from an MCP server.
    
    Agent Cards are standardized metadata documents that describe an agent's
    capabilities, skills, supported modes, and version information. They follow
    the A2A protocol specification and are served at a well-known endpoint.

    Args:
        agent_uri (str): The URI of the agent card as Resource in the MCP Server

    Returns:
        AgentCard: The retrieved Agent Card object.

    Raises:
        httpx.HTTPError: If there is an issue with the HTTP request.
        ValueError: If the response cannot be parsed into an AgentCard.

    Well-known endpoint format: {agent_uri}/.well-known/agent-card.json
    """
    async with Client(MCP_SERVER_URL) as client:
        response = await client.read_resource(agent_uri)
        for content in response:
            if hasattr(content, "text") and content.text:
                try:
                    card_data = json.loads(content.text)
                    return AgentCard.model_validate(card_data)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse Agent Card JSON: {e}")
        raise ValueError("No valid content found for Agent Card.")