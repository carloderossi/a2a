"""
Example 1: Basic LangChain with Different Ollama Models

This example demonstrates:
1. Using different Ollama models with LangChain
2. Switching between models dynamically
3. Comparing outputs from different models
4. Model-specific configurations

Prerequisites:
- Ollama installed and running
- Models pulled: ollama pull qwen3, ollama pull deepseek-r1, ollama pull qwen3
"""

import asyncio
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


# ANSI color codes for terminal output
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

OLLAMA_SLM = "qwen3" # Default Ollama model

def create_ollama_llm(model_name: str, temperature: float = 0.7):
    """
    Create a LangChain ChatOllama instance.
    
    Args:
        model_name (str): Name of the Ollama model (e.g., 'llama3.1', 'mistral')
        temperature (float): Sampling temperature (0.0 to 1.0)
    
    Returns:
        ChatOllama: Configured LangChain Ollama instance
    """
    return ChatOllama(
        model=model_name,
        temperature=temperature,
    )


async def basic_chat_example():
    """Demonstrate basic chat with a single Ollama model."""
    print(f"\n{CYAN}{'='*60}")
    print(f"Basic Chat Example with {OLLAMA_SLM}")
    print(f"{'='*60}{RESET}\n")
    
    # Create LLM instance
    llm = create_ollama_llm(OLLAMA_SLM, temperature=0.7)
    
    # Create messages
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content="Explain quantum computing in 2 sentences.")
    ]
    
    # Get response
    print(f"{YELLOW}Query:{RESET} Explain quantum computing in 2 sentences.")
    response = await llm.ainvoke(messages)
    print(f"{GREEN}Response:{RESET}")
    print(response.content)
    print()


async def compare_models():
    """Compare outputs from different Ollama models."""
    print(f"\n{CYAN}{'='*60}")
    print("Comparing Multiple Ollama Models")
    print(f"{'='*60}{RESET}\n")
    
    # Define models to compare
    models = ["llama3.1", "deepseek-r1", "qwen3"]
    
    # Test query
    query = "What are the key benefits of functional programming?"
    print(f"{YELLOW}Query:{RESET} {query}\n")
    
    # Get responses from each model
    for model_name in models:
        print(f"{CYAN}{'─'*60}")
        print(f"Model: {model_name}")
        print(f"{'─'*60}{RESET}")
        
        try:
            llm = create_ollama_llm(model_name, temperature=0.5)
            
            messages = [
                SystemMessage(content="You are a concise programming expert."),
                HumanMessage(content=query)
            ]
            
            response = await llm.ainvoke(messages)
            print(f"{GREEN}Response:{RESET}")
            print(response.content)
            print()
            
        except Exception as e:
            print(f"Error with {model_name}: {e}\n")


async def temperature_comparison():
    """Demonstrate how temperature affects outputs."""
    print(f"\n{CYAN}{'='*60}")
    print("Temperature Effect Comparison")
    print(f"{'='*60}{RESET}\n")
    
    query = "Write a creative opening line for a sci-fi story."
    temperatures = [0.0, 0.5, 1.0]
    
    print(f"{YELLOW}Query:{RESET} {query}\n")
    
    for temp in temperatures:
        print(f"{CYAN}{'─'*60}")
        print(f"Temperature: {temp}")
        print(f"{'─'*60}{RESET}")
        
        llm = create_ollama_llm(OLLAMA_SLM, temperature=temp)
        
        messages = [
            SystemMessage(content="You are a creative science fiction writer."),
            HumanMessage(content=query)
        ]
        
        response = await llm.ainvoke(messages)
        print(f"{GREEN}Response:{RESET}")
        print(response.content)
        print()


async def streaming_example():
    """Demonstrate streaming responses from Ollama."""
    print(f"\n{CYAN}{'='*60}")
    print("Streaming Response Example")
    print(f"{'='*60}{RESET}\n")
    
    llm = create_ollama_llm(OLLAMA_SLM, temperature=0.7)
    
    query = "List 5 interesting facts about Mars."
    print(f"{YELLOW}Query:{RESET} {query}")
    print(f"{GREEN}Streaming Response:{RESET}")
    
    messages = [HumanMessage(content=query)]
    
    # Stream response
    async for chunk in llm.astream(messages):
        print(chunk.content, end="", flush=True)
    
    print("\n")


async def main():
    """Run all examples."""
    print(f"{GREEN}{'='*60}")
    print("LangChain with Ollama Models - Examples")
    print(f"{'='*60}{RESET}")
    
    # Run examples
    await basic_chat_example()
    await compare_models()
    await temperature_comparison()
    await streaming_example()
    
    print(f"{GREEN}{'='*60}")
    print("All examples completed!")
    print(f"{'='*60}{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
