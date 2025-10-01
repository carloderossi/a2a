# agent_executor.py
import ollama
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import Message, Part
import uuid
import logging

SLM_MODEL = "llama3.1"

# ANSI escape codes
LIGHT_BLUE = '\033[94m'
CYAN = '\033[96m'
GREEN = '\033[92m'
RESET = '\033[0m'
logging.basicConfig(
    #format='[%(filename)s - %(asctime)s] %(message)s',
    format=f'{CYAN}[%(filename)s - %(asctime)s]{RESET} {GREEN}%(levelname)s{RESET} %(message)s',
    datefmt='%m.%d.%Y %H:%M:%S',
    level=logging.INFO
)

class ResearchAgentExecutor:
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        query = context.get_user_input()
        logging.info(f"ResearchAgentExecutor processing query:\n{query}")

        logging.info(f"Sending query to Ollama model: {SLM_MODEL}")
        response = ollama.chat(
            model=SLM_MODEL,
            messages=[{"role": "user", "content": query or ""}],
        )
        logging.info(f"Ollama response: {response}")
        output = response["message"]["content"]
        logging.info(f"Extracted output: {output}")

        # Create a Message and put it directly on the queue
        msg = Message(
            messageId=str(uuid.uuid4()),
            role="agent",
            parts=[Part(kind="text", text=output)],
            final=True,
        )
        logging.info(f"Putting message on event queue: {msg}")
        await event_queue.enqueue_event(msg)

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        pass


class PlannerAgentExecutor:
    async def execute(self, context: RequestContext, event_queue: EventQueue):
        research = context.get_user_input()
        logging.info(f"PlannerAgentExecutor processing research: {research}")
        prompt = f"Based on this research:\n{research}\nCreate a step-by-step plan."
        logging.info(f"PlannerAgentExecutor prompt:\n{prompt}")

        logging.info(f"Sending prompt to Ollama model {SLM_MODEL}...")
        response = ollama.chat(
            model=SLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        logging.info(f"Ollama response: {response}")
        output = response["message"]["content"]

        # Create a Message and put it directly on the queue
        msg = Message(
            messageId=str(uuid.uuid4()),
            role="agent",
            parts=[Part(kind="text", text=output)],
            final=True,
        )
        logging.info(f"Putting message on event queue: {msg}")
        await event_queue.enqueue_event(msg)

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        pass