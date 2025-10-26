"""
Example 3: MCP Integration with LangChain

This example demonstrates:
1. Discovering MCP tools from an MCP server
2. Converting MCP tools to LangChain tools
3. Using MCP tools in LangChain chains
4. Reading MCP resources (like Agent Cards)

Prerequisites:
- MCP server running on http://127.0.0.1:8080/mcp
- Ollama with llama3.1 model
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.mcp_tools import MCPToolWrapper, get_mcp_tools_for_langchain, read_mcp_resource
from utils.a2a_utils import resolve_agent_card, MCP_SERVER_URL
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage


# ANSI color codes
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'


async def list_mcp_tools_example():
    """List all available MCP tools."""
    print(f"\n{CYAN}{'='*60}")
    print("List Available MCP Tools")
    print(f"{'='*60}{RESET}\n")
    
    try:
        wrapper = MCPToolWrapper(MCP_SERVER_URL)
        tools = await wrapper.list_tools()
        
        print(f"{GREEN}Found {len(tools)} MCP tools:{RESET}\n")
        
        for i, tool in enumerate(tools, 1):
            print(f"{YELLOW}{i}. {tool.name}{RESET}")
            print(f"   Description: {tool.description}")
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                print(f"   Input Schema: {tool.inputSchema.get('properties', {}).keys()}")
            print()
            
    except Exception as e:
        print(f"{RED}Error connecting to MCP server: {e}{RESET}")
        print(f"{YELLOW}Make sure MCP server is running on {MCP_SERVER_URL}{RESET}")


async def call_mcp_tool_example():
    """Demonstrate calling an MCP tool directly."""
    print(f"\n{CYAN}{'='*60}")
    print("Call MCP Tool Directly")
    print(f"{'='*60}{RESET}\n")
    
    try:
        wrapper = MCPToolWrapper(MCP_SERVER_URL)
        
        # List available tools first
        tools = await wrapper.list_tools()
        
        if not tools:
            print(f"{YELLOW}No tools available on MCP server{RESET}")
            return
        
        # Use the first available tool as an example
        tool = tools[0]
        print(f"{YELLOW}Testing tool: {tool.name}{RESET}")
        print(f"Description: {tool.description}\n")
        
        # Note: You'll need to adjust arguments based on the actual tool
        # This is a generic example
        print(f"{GREEN}Tool call would go here with appropriate arguments{RESET}")
        print(f"Example: result = await wrapper.call_tool('{tool.name}', {{...}})")
        print()
        
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")


async def mcp_tools_in_langchain():
    """Use MCP tools with a LangChain agent."""
    print(f"\n{CYAN}{'='*60}")
    print("MCP Tools in LangChain")
    print(f"{'='*60}{RESET}\n")
    
    try:
        # Get MCP tools as LangChain tools
        print(f"{YELLOW}Converting MCP tools to LangChain format...{RESET}")
        mcp_tools = await get_mcp_tools_for_langchain(MCP_SERVER_URL)
        
        if not mcp_tools:
            print(f"{YELLOW}No MCP tools available{RESET}")
            return
        
        print(f"{GREEN}Successfully converted {len(mcp_tools)} MCP tools!{RESET}\n")
        
        # Display converted tools
        for tool in mcp_tools:
            print(f"  - {tool.name}: {tool.description}")
        print()
        
        # Create LangChain LLM
        llm = ChatOllama(model="llama3.1", temperature=0.7)
        
        # Note: Tool binding and agent creation would go here
        # The exact implementation depends on which MCP tools are available
        
        print(f"{GREEN}MCP tools are now ready to use with LangChain agents!{RESET}")
        print(f"{YELLOW}To use them, bind them to an LLM: llm.bind_tools(mcp_tools){RESET}")
        print()
        
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")


async def read_agent_card_from_mcp():
    """Read an Agent Card from MCP as a resource."""
    print(f"\n{CYAN}{'='*60}")
    print("Read Agent Card from MCP")
    print(f"{'='*60}{RESET}\n")
    
    try:
        # Try to read a research agent card
        agent_uri = "agent://cdr.Research_Agent"
        
        print(f"{YELLOW}Reading Agent Card for: {agent_uri}{RESET}\n")
        
        card = await resolve_agent_card(agent_uri, debug=True)
        
        print(f"{GREEN}Agent Card Retrieved:{RESET}")
        print(f"  Name: {card.name}")
        print(f"  Description: {card.description}")
        print(f"  URL: {card.url}")
        print(f"  Version: {card.version}")
        
        if hasattr(card, 'skills') and card.skills:
            print(f"  Skills:")
            for skill in card.skills:
                print(f"    - {skill.name}: {skill.description}")
        
        # Handle optional attributes with defaults
        input_modes = getattr(card, 'defaultInputModes', None) or getattr(card, 'default_input_modes', ['text'])
        output_modes = getattr(card, 'defaultOutputModes', None) or getattr(card, 'default_output_modes', ['text'])
        
        print(f"\n  Input Modes: {', '.join(input_modes)}")
        print(f"  Output Modes: {', '.join(output_modes)}")
        
        # Show capabilities if available
        if hasattr(card, 'capabilities'):
            print(f"  Capabilities: {card.capabilities}")
        
        print()
        
    except AttributeError as e:
        print(f"{RED}Error: Missing expected attribute - {e}{RESET}")
        print(f"{YELLOW}Agent card structure may be different than expected{RESET}")
        print(f"{YELLOW}Available attributes: {dir(card) if 'card' in locals() else 'N/A'}{RESET}")
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")
        print(f"{YELLOW}Make sure the agent is registered with MCP server{RESET}")


async def mcp_resource_in_chain():
    """Use MCP resources in a LangChain chain."""
    print(f"\n{CYAN}{'='*60}")
    print("MCP Resources in LangChain Chain")
    print(f"{'='*60}{RESET}\n")
    
    try:
        # Read agent information from MCP
        agent_uri = "agent://cdr.Research_Agent"
        
        print(f"{YELLOW}Reading agent information...{RESET}")
        card = await resolve_agent_card(agent_uri)
        
        # Create a prompt that uses this information
        llm = ChatOllama(model="llama3.1", temperature=0.7)
        
        prompt = f"""
        Based on this agent information:
        
        Agent: {card.name}
        Description: {card.description}
        Skills: {', '.join([s.name for s in card.skills])}
        
        Explain what this agent can do and when to use it.
        """
        
        print(f"{YELLOW}Generating explanation...{RESET}\n")
        
        messages = [HumanMessage(content=prompt)]
        response = await llm.ainvoke(messages)
        
        print(f"{GREEN}LangChain Response:{RESET}")
        print(response.content)
        print()
        
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")


async def mcp_tool_wrapper_demo():
    """Demonstrate the MCPToolWrapper class features."""
    print(f"\n{CYAN}{'='*60}")
    print("MCPToolWrapper Class Demo")
    print(f"{'='*60}{RESET}\n")
    
    try:
        wrapper = MCPToolWrapper(MCP_SERVER_URL)
        
        # List tools
        print(f"{YELLOW}1. Listing MCP tools...{RESET}")
        tools = await wrapper.list_tools()
        print(f"   Found: {[t.name for t in tools]}\n")
        
        # Convert to LangChain tools
        print(f"{YELLOW}2. Converting to LangChain tools...{RESET}")
        langchain_tools = [wrapper.create_langchain_tool(t) for t in tools]
        print(f"   Converted: {[t.name for t in langchain_tools]}\n")
        
        # Get all tools at once
        print(f"{YELLOW}3. Getting all tools in one call...{RESET}")
        all_tools = await wrapper.get_all_langchain_tools()
        print(f"   Retrieved: {len(all_tools)} tools\n")
        
        print(f"{GREEN}MCPToolWrapper demo completed!{RESET}")
        print()
        
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")


async def main():
    """Run all MCP integration examples."""
    print(f"{GREEN}{'='*60}")
    print("LangChain with MCP Integration - Examples")
    print(f"{'='*60}{RESET}")
    
    # Check MCP server connectivity
    print(f"\n{YELLOW}Checking MCP server at {MCP_SERVER_URL}...{RESET}")
    
    try:
        # Try a simple connection test
        wrapper = MCPToolWrapper(MCP_SERVER_URL)
        tools = await wrapper.list_tools()
        print(f"{GREEN}✓ MCP server is accessible{RESET}")
        print(f"{GREEN}✓ Found {len(tools)} tools{RESET}\n")
        
        # Run examples
        await list_mcp_tools_example()
        await call_mcp_tool_example()
        await mcp_tools_in_langchain()
        await read_agent_card_from_mcp()
        await mcp_resource_in_chain()
        await mcp_tool_wrapper_demo()
        
    except Exception as e:
        print(f"{RED}✗ Cannot connect to MCP server{RESET}")
        print(f"{YELLOW}Please ensure MCP server is running on {MCP_SERVER_URL}{RESET}")
        print(f"{RED}Error details: {type(e).__name__}: {e}{RESET}\n")
        import traceback
        traceback.print_exc()
    
    print(f"{GREEN}{'='*60}")
    print("All examples completed!")
    print(f"{'='*60}{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
