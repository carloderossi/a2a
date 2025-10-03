# async_test_client.py
import httpx
import json
import uuid 
from typing import Any
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

# URLs for the two agent servers
RESEARCH_URL = "http://localhost:9001/"
PLANNER_URL = "http://localhost:9002/"

def get_client(card:AgentCard, request_timeout=360.0, connect_timeout=60.0) -> A2AClient:
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
    client, httpx_client = get_client(card)
    print("A2AClient initialized via ClientFactory")
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
            
async def fetch_agent_card(base_url):
    # Create an instance of httpx.Client
    async with httpx.AsyncClient() as httpx_client:
        # Initialize A2ACardResolver with the client
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )
        card = await resolver.get_agent_card()
        #print(card.model_dump_json(indent=2))
        return card

def extract_text(resp) -> str | None:
    data = resp.model_dump()
    try:
        return data["result"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return None

async def main():
    # Fetch metadata from both agents to verify they're available and properly configured
    # This demonstrates the A2A service discovery pattern using Agent Cards
    research_card = await fetch_agent_card(RESEARCH_URL)
    planner_card = await fetch_agent_card(PLANNER_URL)
    print(f"\nResearch AgentCard: {research_card.name}")
    print(research_card.model_dump_json(indent=2))
    print(f"\nPlanner AgentCard: {planner_card.name}")
    print(planner_card.model_dump_json(indent=2))

    # Define the initial user query
    query = "Summarize the latest approaches to reinforcement learning exploration."
    print(f"\n[User Query]\n {query}")
    
    # Stage 1: Send query to Research Agent for information gathering
    print("Sending query to Research agent...")
    research_result = await run_messagev2(research_card, query)
    print(f"\n[Research Result]\n{research_result}\n")    

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())