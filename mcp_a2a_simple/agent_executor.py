"""
Agent Executor Module

This module defines executor classes for Research and Planner agents in an Agent-to-Agent (A2A)
communication system. Each executor handles the core logic of processing requests and generating
responses using the Ollama LLM.

A2A Protocol Reference:
    Official Documentation: https://a2a-protocol.org/
    GitHub Repository: https://github.com/google-a2a/A2A
    Protocol Specification: https://a2a-protocol.org/latest/spec/
"""

import ollama
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import Message, Part
import uuid
import logging

# Ollama model configuration - using LLaMA 3.1 as the default SLM (Small Language Model)
SLM_MODEL = "qwen3"

# ANSI escape codes for colored terminal output
LIGHT_BLUE = '\033[94m'
CYAN = '\033[96m'
GREEN = '\033[92m'
RESET = '\033[0m'

# Configure logging with colored output for better readability
logging.basicConfig(
    format=f'{CYAN}[%(filename)s - %(asctime)s]{RESET} {GREEN}%(levelname)s{RESET} %(message)s',
    datefmt='%m.%d.%Y %H:%M:%S',
    level=logging.INFO
)


class ResearchAgentExecutor:
    """
    Executor for the Research Agent.
    
    This class handles research-oriented queries by sending them to the Ollama LLM
    and returning summarized or gathered information. It's designed to be used
    within the A2A framework for agent-to-agent communication.
    
    A2A Concepts Used:
        - Message: Structured communication format (see A2A spec section on Messages)
        - EventQueue: Asynchronous message delivery mechanism
        - RequestContext: Container for request data and agent state
    """
    
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """
        Execute a research query using the Ollama LLM.
        
        This method:
        1. Extracts the user query from the request context
        2. Sends the query to the Ollama model
        3. Processes the response
        4. Enqueues the result as a Message event for downstream consumption
        
        Args:
            context (RequestContext): Contains the incoming request data and user input
            event_queue (EventQueue): Queue for publishing response messages to consumers
        
        Returns:
            None: Results are published to the event queue asynchronously
        
        Note:
            The Message format follows the A2A protocol specification for interoperability.
            See: https://a2a-protocol.org/latest/spec/#messages
        """
        # Extract the user's query from the request context
        query = context.get_user_input()
        logging.info(f"ResearchAgentExecutor processing query:\n{query}")

        # Send the query to Ollama for processing
        logging.info(f"Sending query to Ollama model: {SLM_MODEL}")
        response = ollama.chat(
            model=SLM_MODEL,
            messages=[{"role": "user", "content": query or ""}],
        )
        logging.info(f"Ollama response: {response}")
        
        # Extract the text content from the model's response
        output = response["message"]["content"]
        logging.info(f"Extracted output: {output}")

        # Create a Message object following the A2A protocol specification
        # This message will be consumed by the client or forwarded to another agent
        msg = Message(
            messageId=str(uuid.uuid4()),  # Unique identifier for message tracking
            role="agent",  # Indicates this message is from an agent, not a user
            parts=[Part(kind="text", text=output)],  # Message content as text part
            final=True,  # Indicates this is the final message (not streaming)
        )
        logging.info(f"Putting message on event queue: {msg}")
        
        # Enqueue the message for asynchronous delivery
        await event_queue.enqueue_event(msg)

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """
        Handle cancellation of ongoing research tasks.
        
        Currently not implemented as Ollama calls are synchronous and cannot be
        easily cancelled mid-execution. Future implementations could add timeout
        handling or task interruption logic.
        
        Args:
            context (RequestContext): Request context containing cancellation details
            event_queue (EventQueue): Event queue for publishing cancellation notifications
        
        Note:
            Task cancellation is part of the A2A task lifecycle management.
            See: https://a2a-protocol.org/latest/spec/#task-lifecycle
        """
        pass


class PlannerAgentExecutor:
    """
    Executor for the Planner Agent.
    
    This class takes research results and generates structured, step-by-step plans
    using the Ollama LLM. It's designed to work in a pipeline where research output
    is transformed into actionable plans.
    
    A2A Concepts Used:
        - Message: Protocol-standard message format for responses
        - Agent chaining: Receiving output from one agent as input to another
    """
    
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """
        Generate a structured plan based on research input.
        
        This method:
        1. Receives research data from the context
        2. Constructs a planning prompt that instructs the LLM to create a plan
        3. Sends the prompt to Ollama
        4. Returns the generated plan via the event queue
        
        Args:
            context (RequestContext): Contains research results from the previous agent
            event_queue (EventQueue): Queue for publishing the generated plan
        
        Returns:
            None: Results are published to the event queue asynchronously
        
        Note:
            This demonstrates A2A agent chaining where the output of one agent
            becomes the input to another agent in a workflow.
        """
        # Get the research results from the upstream agent
        research = context.get_user_input()
        logging.info(f"PlannerAgentExecutor processing research: {research}")
        
        # Construct a prompt that instructs the LLM to create a structured plan
        prompt = f"Based on this research:\n{research}\nCreate a step-by-step plan."
        logging.info(f"PlannerAgentExecutor prompt:\n{prompt}")

        # Send the planning prompt to Ollama
        logging.info(f"Sending prompt to Ollama model {SLM_MODEL}...")
        response = ollama.chat(
            model=SLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        logging.info(f"Ollama response: {response}")
        
        # Extract the generated plan from the model's response
        output = response["message"]["content"]

        # Create a Message object with the planning results
        msg = Message(
            messageId=str(uuid.uuid4()),  # Unique message identifier
            role="agent",  # Indicates agent-generated content
            parts=[Part(kind="text", text=output)],  # The generated plan
            final=True,  # Final message in this interaction
        )
        logging.info(f"Putting message on event queue: {msg}")
        
        # Publish the plan to the event queue
        await event_queue.enqueue_event(msg)

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """
        Handle cancellation of ongoing planning tasks.
        
        Currently not implemented. Future versions could add support for
        interrupting long-running planning operations.
        
        Args:
            context (RequestContext): Request context with cancellation details
            event_queue (EventQueue): Event queue for cancellation notifications
        """
        pass