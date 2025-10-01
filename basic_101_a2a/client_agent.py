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

# URLs for the two agent servers
RESEARCH_URL = "http://localhost:9001/"
PLANNER_URL = "http://localhost:9002/"

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


def fetch_agent_card(base_url):
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
    resp = requests.get(f"{base_url}.well-known/agent-card.json", timeout=5)
    resp.raise_for_status()
    return resp.json()


def rpc_call(base_url, method, params):
    """
    Make a JSON-RPC 2.0 call to an A2A agent server.
    
    This function implements the JSON-RPC protocol used by A2A agents for
    communication. It handles request formatting, error checking, and
    response parsing according to both JSON-RPC 2.0 and A2A specifications.
    
    Args:
        base_url (str): Base URL of the agent server
        method (str): RPC method name (e.g., "message/send", "tasks/get")
        params (dict): Parameters to pass to the RPC method
    
    Returns:
        dict: The result field from the JSON-RPC response
    
    Raises:
        RuntimeError: If the agent returns an error in the response
        requests.HTTPError: If the HTTP request fails
    
    A2A Reference:
        JSON-RPC Methods: https://a2a-protocol.org/latest/spec/#json-rpc-methods
        Common methods include: message/send, message/stream, tasks/get, tasks/list
    """
    # Construct a JSON-RPC 2.0 request payload
    payload = {
        "jsonrpc": "2.0",  # Protocol version
        "id": str(uuid.uuid4()),  # Unique request identifier for matching responses
        "method": method,  # The RPC method to invoke
        "params": params,  # Method-specific parameters
    }
    
    # Send the RPC request with a generous timeout (5 minutes for long-running tasks)
    resp = requests.post(base_url, json=payload, timeout=300)
    logging.info(f"RPC call response status: {resp.status_code}")
    resp.raise_for_status()
    
    # Parse the JSON-RPC response
    data = resp.json()
    logging.info(f"RPC call response: {data}")
    
    # Check for RPC-level errors (distinct from HTTP errors)
    if "error" in data:
        raise RuntimeError(f"Agent error: {data['error']}")
    
    return data["result"]


def run_message(base_url, text):
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
    # Step 1: Send the message to the agent using the message/send RPC method
    result = rpc_call(base_url, "message/send", {
        "message": {
            "role": "user",  # Message is from the user/client
            "parts": [{"kind": "text", "text": text}],  # Message content
            "messageId": str(uuid.uuid4())  # Unique message identifier
        }
    })
    
    # Handle immediate response (synchronous mode)
    if result.get("kind") == "message":
        # Extract all text parts from the message
        text_parts = [p["text"] for p in result.get("parts", []) if p["kind"] == "text"]
        return "\n".join(text_parts)

    # Handle async task response (streaming mode)
    if result.get("kind") == "task":
        return result["id"]
    
    # Extract task ID for polling
    task_id = result.get("id")  
    if not task_id:
        raise RuntimeError(f"Unexpected result: {result}")

    # Step 2: Poll the task endpoint until completion
    # This implements the A2A task lifecycle polling pattern
    while True:
        # Query the task status using the tasks/get RPC method
        status = rpc_call(base_url, "tasks/get", {"id": task_id})
        
        # Check task state (handles both "status" and "state" field names)
        state = status.get("status") or status.get("state")
        
        if state in ("succeeded", "completed"):
            # Task completed successfully, return the output
            return status.get("result", {}).get("output")
        elif state in ("failed", "error"):
            # Task failed, raise an error with details
            raise RuntimeError(f"Task {task_id} failed: {status}")
        
        # Wait 1 second before polling again to avoid overwhelming the server
        time.sleep(1)


def main():
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
    research_card = fetch_agent_card(RESEARCH_URL)
    planner_card = fetch_agent_card(PLANNER_URL)
    
    logging.info(f"Research AgentCard: {research_card['name']}")
    print(f"{research_card}\n")
    logging.info(f"Planner AgentCard: {planner_card['name']}")
    print(f"{planner_card}\n")

    # Define the initial user query
    query = "Summarize the latest approaches to reinforcement learning exploration."
    logging.info(f"\n[User Query]\n {query}")
    
    # Stage 1: Send query to Research Agent for information gathering
    logging.info("Sending query to Research agent...")
    research_result = run_message(RESEARCH_URL, query)
    logging.info(f"\n[Research Result]\n{research_result}\n")

    # Stage 2: Send research results to Planner Agent for plan generation
    # This demonstrates agent chaining - output from one agent becomes input to another
    logging.info("Sending research result to Planner agent...")
    plan_result = run_message(PLANNER_URL, research_result)
    logging.info(f"\n[Planner Result]\n{plan_result}\n")

    logging.info("Done.")


if __name__ == "__main__":
    main()


"""
Commented out code from the original file:
This section was likely used for debugging or an alternative logging approach.

logging.info("Research AgentCard:", research_card["name"])
logging.info(research_card)
logging.info("Planner AgentCard:", planner_card["name"])
logging.info(planner_card)
"""