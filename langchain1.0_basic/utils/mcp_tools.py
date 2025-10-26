"""
MCP Tools Integration for LangChain

This module provides utilities for integrating MCP (Model Context Protocol) tools
with LangChain. It includes functions to discover MCP tools and convert them into
LangChain-compatible tool interfaces.
"""

from typing import Any, Callable, Optional
from fastmcp import Client
from mcp.types import Tool as MCPTool, TextContent
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import json

MCP_SERVER_URL = "http://127.0.0.1:8080/mcp"


class MCPToolWrapper:
    """
    Wrapper class that converts MCP tools into LangChain-compatible tools.
    
    This enables seamless integration of MCP tools within LangChain chains and agents.
    """
    
    def __init__(self, mcp_server_url: str = MCP_SERVER_URL):
        """
        Initialize the MCP tool wrapper.
        
        Args:
            mcp_server_url (str): URL of the MCP server
        """
        self.mcp_server_url = mcp_server_url
        self._client: Optional[Client] = None
    
    async def list_tools(self) -> list[MCPTool]:
        """
        List all available tools from the MCP server.
        
        Returns:
            list[MCPTool]: List of available MCP tools
            
        Example:
            >>> wrapper = MCPToolWrapper()
            >>> tools = await wrapper.list_tools()
            >>> for tool in tools:
            ...     print(f"{tool.name}: {tool.description}")
        """
        async with Client(self.mcp_server_url) as client:
            tools_response = await client.list_tools()
            # Handle both response types: object with .tools or direct list
            if isinstance(tools_response, list):
                return tools_response
            elif hasattr(tools_response, 'tools'):
                return tools_response.tools
            else:
                return []
    
    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Call an MCP tool with the given arguments.
        
        Args:
            tool_name (str): Name of the tool to call
            arguments (dict): Arguments to pass to the tool
            
        Returns:
            Any: The tool's response
            
        Example:
            >>> wrapper = MCPToolWrapper()
            >>> result = await wrapper.call_tool("calculator", {"operation": "add", "a": 5, "b": 3})
            >>> print(result)
        """
        async with Client(self.mcp_server_url) as client:
            result = await client.call_tool(tool_name, arguments)
            
            # Handle different result types
            if not result:
                return None
            
            # If result is a list, process first item
            if isinstance(result, list) and len(result) > 0:
                first_item = result[0]
                if isinstance(first_item, TextContent):
                    return first_item.text
                elif hasattr(first_item, 'text'):
                    return first_item.text
                else:
                    return str(first_item)
            
            # If result has text attribute directly
            if hasattr(result, 'text'):
                return result.text
            
            # Return as string
            return str(result)
    
    def create_langchain_tool(
        self,
        mcp_tool: MCPTool,
        handle_tool_error: bool = True
    ) -> StructuredTool:
        """
        Convert an MCP tool into a LangChain StructuredTool.
        
        Args:
            mcp_tool (MCPTool): The MCP tool to convert
            handle_tool_error (bool): Whether to handle errors gracefully
            
        Returns:
            StructuredTool: A LangChain-compatible tool
            
        Example:
            >>> wrapper = MCPToolWrapper()
            >>> tools = await wrapper.list_tools()
            >>> langchain_tools = [wrapper.create_langchain_tool(t) for t in tools]
        """
        
        async def tool_func(**kwargs) -> str:
            """Execute the MCP tool with the given arguments."""
            try:
                result = await self.call_tool(mcp_tool.name, kwargs)
                return str(result) if result else "No result returned"
            except Exception as e:
                if handle_tool_error:
                    return f"Error calling tool {mcp_tool.name}: {str(e)}"
                raise
        
        # Create LangChain tool using StructuredTool
        return StructuredTool.from_function(
            name=mcp_tool.name,
            description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
            func=tool_func,
            coroutine=tool_func,  # Support async execution
        )
    
    async def get_all_langchain_tools(
        self,
        handle_tool_error: bool = True
    ) -> list[StructuredTool]:
        """
        Get all MCP tools as LangChain tools.
        
        Args:
            handle_tool_error (bool): Whether to handle errors gracefully
            
        Returns:
            list[StructuredTool]: List of LangChain-compatible tools
            
        Example:
            >>> wrapper = MCPToolWrapper()
            >>> tools = await wrapper.get_all_langchain_tools()
            >>> # Use tools in LangChain chains
        """
        mcp_tools = await self.list_tools()
        return [
            self.create_langchain_tool(tool, handle_tool_error)
            for tool in mcp_tools
        ]


async def get_mcp_tools_for_langchain(
    mcp_server_url: str = MCP_SERVER_URL,
    handle_tool_error: bool = True
) -> list[StructuredTool]:
    """
    Convenience function to get all MCP tools as LangChain tools.
    
    Args:
        mcp_server_url (str): URL of the MCP server
        handle_tool_error (bool): Whether to handle errors gracefully
        
    Returns:
        list[StructuredTool]: List of LangChain-compatible tools
        
    Example:
        >>> tools = await get_mcp_tools_for_langchain()
        >>> # Use tools in LangChain chains
    """
    wrapper = MCPToolWrapper(mcp_server_url)
    return await wrapper.get_all_langchain_tools(handle_tool_error)


async def read_mcp_resource(resource_uri: str) -> str:
    """
    Read a resource from the MCP server.
    
    Args:
        resource_uri (str): URI of the resource to read
        
    Returns:
        str: The resource content as text
        
    Example:
        >>> content = await read_mcp_resource("agent://cdr.Research_Agent")
        >>> print(content)
    """
    async with Client(MCP_SERVER_URL) as client:
        response = await client.read_resource(resource_uri)
        
        # Handle list response
        if isinstance(response, list):
            for content in response:
                if hasattr(content, "text") and content.text:
                    return content.text
        # Handle single object response
        elif hasattr(response, "text") and response.text:
            return response.text
        
        return ""


# Example MCP tool schema for custom tools
class CalculatorInput(BaseModel):
    """Input schema for a calculator tool."""
    operation: str = Field(description="The operation to perform: add, subtract, multiply, divide")
    a: float = Field(description="First number")
    b: float = Field(description="Second number")


class SearchInput(BaseModel):
    """Input schema for a search tool."""
    query: str = Field(description="The search query")
    num_results: int = Field(default=5, description="Number of results to return")
