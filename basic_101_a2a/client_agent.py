import requests
import uuid
import time
import logging

RESEARCH_URL = "http://localhost:9001/"
PLANNER_URL = "http://localhost:9002/"

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

def fetch_agent_card(base_url):
    resp = requests.get(f"{base_url}.well-known/agent-card.json", timeout=5)
    resp.raise_for_status()
    return resp.json()

def rpc_call(base_url, method, params):
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": params,
    }
    resp = requests.post(base_url, json=payload, timeout=300)
    logging.info(f"RPC call response status: {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()
    logging.info(f"RPC call response: {data}")
    if "error" in data:
        raise RuntimeError(f"Agent error: {data['error']}")
    return data["result"]

def run_message(base_url, text):
    # 1. Send message
    result = rpc_call(base_url, "message/send", {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": text}],
            "messageId": str(uuid.uuid4())
        }
    })
    # If the result is a Message
    if result.get("kind") == "message":
        # Access the text parts directly
        text_parts = [p["text"] for p in result.get("parts", []) if p["kind"] == "text"]
        return "\n".join(text_parts)

    # If the result is a Task (streaming mode)
    if result.get("kind") == "task":
        return result["id"]
    task_id = result.get("id")  
    if not task_id:
        raise RuntimeError(f"Unexpected result: {result}")

    # 2. Poll until done
    while True:
        status = rpc_call(base_url, "tasks/get", {"id": task_id})
        state = status.get("status") or status.get("state")
        if state in ("succeeded", "completed"):
            return status.get("result", {}).get("output")
        elif state in ("failed", "error"):
            raise RuntimeError(f"Task {task_id} failed: {status}")
        time.sleep(1)

def main():
    research_card = fetch_agent_card(RESEARCH_URL)
    planner_card = fetch_agent_card(PLANNER_URL)
    logging.info(f"Research AgentCard: {research_card['name']}")
    print(f"{research_card}\n")
    logging.info(f"Planner AgentCard: {planner_card['name']}")
    print(f"{planner_card}\n")

    query = "Summarize the latest approaches to reinforcement learning exploration."
    logging.info(f"\n[User Query]\n {query}")
    logging.info("Sending query to Research agent...")
    research_result = run_message(RESEARCH_URL, query)
    logging.info(f"\n[Research Result]\n{research_result}\n")

    logging.info("Sending research result to Planner agent...")
    plan_result = run_message(PLANNER_URL, research_result)
    logging.info(f"\n[Planner Result]\n{plan_result}\n")

    logging.info("Done.")

if __name__ == "__main__":
    main()

""" 
    logging.info("Research AgentCard:", research_card["name"])
    logging.info(research_card)
    logging.info("Planner AgentCard:", planner_card["name"])
    logging.info(planner_card) """