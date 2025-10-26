"""
Utilities for LangChain with MCP and A2A integration.
"""

from .a2a_utils import (
    register_agentcard,
    remove_agentcard,
    resolve_agent_card,
    get_a2a_client,
    extract_text_from_response,
    MCP_SERVER_URL,
    LANGCHAIN_RESEARCH_AGENT,
    LANGCHAIN_PLANNER_AGENT,
)

from .mcp_tools import (
    MCPToolWrapper,
    get_mcp_tools_for_langchain,
    read_mcp_resource,
)

__all__ = [
    # A2A utilities
    "register_agentcard",
    "remove_agentcard",
    "resolve_agent_card",
    "get_a2a_client",
    "extract_text_from_response",
    "MCP_SERVER_URL",
    "LANGCHAIN_RESEARCH_AGENT",
    "LANGCHAIN_PLANNER_AGENT",
    # MCP utilities
    "MCPToolWrapper",
    "get_mcp_tools_for_langchain",
    "read_mcp_resource",
]
