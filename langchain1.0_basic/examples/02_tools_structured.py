"""
Example 2: Tool Calls and Structured Outputs with LangChain

This example demonstrates:
1. Creating custom tools for LangChain
2. Tool binding and invocation
3. Structured outputs using Pydantic models
4. Simple tool calling with Ollama

Prerequisites:
- Ollama installed with qwen3 model
"""

import asyncio
from typing import List
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage


# ANSI color codes
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'

OLLAMA_SLM = "qwen3" # Default Ollama model

# ============================================================================
# PART 1: Define Custom Tools
# ============================================================================

@tool
def calculator(operation: str, a: float, b: float) -> float:
    """
    Perform basic arithmetic operations.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number
    
    Returns:
        float: Result of the operation
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float('inf')
    }
    
    if operation not in operations:
        return f"Unknown operation: {operation}"
    
    result = operations[operation](a, b)
    return result


@tool
def word_counter(text: str) -> dict:
    """
    Count words, characters, and sentences in a text.
    
    Args:
        text: The text to analyze
    
    Returns:
        dict: Dictionary with counts
    """
    words = len(text.split())
    chars = len(text)
    sentences = text.count('.') + text.count('!') + text.count('?')
    
    return {
        "words": words,
        "characters": chars,
        "sentences": sentences
    }


@tool
def list_generator(category: str, count: int) -> List[str]:
    """
    Generate a list of items in a given category.
    
    Args:
        category: Category of items (e.g., 'colors', 'animals', 'countries')
        count: Number of items to generate
    
    Returns:
        List[str]: List of items
    """
    categories = {
        "colors": ["red", "blue", "green", "yellow", "purple", "orange"],
        "animals": ["dog", "cat", "elephant", "lion", "tiger", "bear"],
        "countries": ["USA", "China", "India", "Brazil", "Russia", "Japan"],
    }
    
    items = categories.get(category.lower(), ["item1", "item2", "item3"])
    return items[:min(count, len(items))]


# ============================================================================
# PART 2: Structured Output Models
# ============================================================================

class SummaryOutput(BaseModel):
    """Structured output for text summarization."""
    title: str = Field(description="A short title for the summary")
    key_points: List[str] = Field(description="List of key points (3-5 items)")
    word_count: int = Field(description="Approximate word count")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")


class ResearchPlan(BaseModel):
    """Structured output for a research plan."""
    topic: str = Field(description="Research topic")
    objectives: List[str] = Field(description="Research objectives")
    methodology: str = Field(description="Research methodology")
    timeline: str = Field(description="Estimated timeline")


class CodeAnalysis(BaseModel):
    """Structured output for code analysis."""
    language: str = Field(description="Programming language")
    complexity: str = Field(description="Code complexity: low, medium, or high")
    suggestions: List[str] = Field(description="Improvement suggestions")
    estimated_lines: int = Field(description="Estimated lines of code")


# ============================================================================
# PART 3: Examples
# ============================================================================

async def tool_calling_example():
    """Demonstrate basic tool calling with LangChain 1.0."""
    print(f"\n{CYAN}{'='*60}")
    print("Tool Calling Example")
    print(f"{'='*60}{RESET}\n")
    
    # Create LLM
    llm = ChatOllama(model=OLLAMA_SLM, temperature=0)
    
    # Define tools
    tools = [calculator, word_counter, list_generator]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Test query
    query = "What is 25 multiplied by 4?"
    print(f"{YELLOW}Query:{RESET} {query}\n")
    
    messages = [
        SystemMessage(content="You are a helpful assistant with access to tools."),
        HumanMessage(content=query)
    ]
    response = await llm_with_tools.ainvoke(messages)
    
    print(f"{GREEN}Response:{RESET}")
    print(response.content)
    
    # Check for tool calls
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"\n{GREEN}Tool Calls Detected:{RESET}")
        for i, tool_call in enumerate(response.tool_calls, 1):
            print(f"  {i}. Tool: {tool_call.get('name', 'unknown')}")
            print(f"     Args: {tool_call.get('args', {})}")
            
            # Execute the tool manually
            tool_name = tool_call.get('name')
            tool_args = tool_call.get('args', {})
            
            if tool_name == 'calculator':
                result = calculator.invoke(tool_args)
                print(f"     Result: {result}")
    else:
        print(f"\n{YELLOW}Note: Tool calls may not be supported by this model{RESET}")
    print()


async def manual_tool_usage():
    """Demonstrate manual tool usage without agent."""
    print(f"\n{CYAN}{'='*60}")
    print("Manual Tool Usage Example")
    print(f"{'='*60}{RESET}\n")
    
    print(f"{YELLOW}Using calculator tool directly:{RESET}")
    result = calculator.invoke({"operation": "multiply", "a": 25, "b": 4})
    print(f"  25 × 4 = {result}\n")
    
    print(f"{YELLOW}Using word_counter tool:{RESET}")
    text = "The quick brown fox jumps over the lazy dog."
    result = word_counter.invoke({"text": text})
    print(f"  Text: {text}")
    print(f"  Analysis: {result}\n")
    
    print(f"{YELLOW}Using list_generator tool:{RESET}")
    result = list_generator.invoke({"category": "colors", "count": 4})
    print(f"  Colors: {result}\n")


async def structured_output_example():
    """Demonstrate structured outputs using Pydantic models."""
    print(f"\n{CYAN}{'='*60}")
    print("Structured Output Example")
    print(f"{'='*60}{RESET}\n")
    
    # Create LLM with structured output
    llm = ChatOllama(model=OLLAMA_SLM, temperature=0)
    
    # Define structured output schema
    structured_llm = llm.with_structured_output(SummaryOutput)
    
    # Test text
    text = """
    Machine learning is revolutionizing how we approach problem-solving in technology.
    It enables computers to learn from data and improve their performance over time
    without being explicitly programmed. Deep learning, a subset of machine learning,
    uses neural networks with multiple layers to process complex patterns. These
    technologies are now being applied in various fields including healthcare,
    finance, and autonomous vehicles.
    """
    
    query = f"Summarize this text:\n{text}"
    print(f"{YELLOW}Query:{RESET} Summarize the text")
    print(f"{YELLOW}Text:{RESET} {text[:100]}...\n")
    
    # Get structured response
    response: SummaryOutput = await structured_llm.ainvoke(query)
    
    print(f"{GREEN}Structured Output:{RESET}")
    print(f"  Title: {response.title}")
    print(f"  Sentiment: {response.sentiment}")
    print(f"  Word Count: {response.word_count}")
    print(f"  Key Points:")
    for i, point in enumerate(response.key_points, 1):
        print(f"    {i}. {point}")
    print()


async def llm_guided_tool_usage():
    """Let LLM guide which tools to use."""
    print(f"\n{CYAN}{'='*60}")
    print("LLM-Guided Tool Usage")
    print(f"{'='*60}{RESET}\n")
    
    llm = ChatOllama(model=OLLAMA_SLM, temperature=0)
    
    # Create a prompt that asks LLM to decide which tool to use
    tools_info = """
Available tools:
1. calculator(operation, a, b) - performs arithmetic
2. word_counter(text) - counts words, chars, sentences
3. list_generator(category, count) - generates lists

Which tool should be used for: "Count the words in 'Hello World'"?
Respond with: tool_name and arguments in JSON format.
"""
    
    print(f"{YELLOW}Query:{RESET} Count words in 'Hello World'\n")
    
    messages = [
        SystemMessage(content="You are a helpful assistant that recommends tools."),
        HumanMessage(content=tools_info)
    ]
    
    response = await llm.ainvoke(messages)
    print(f"{GREEN}LLM Recommendation:{RESET}")
    print(response.content)
    print()
    
    # Execute the recommended tool
    print(f"{GREEN}Executing word_counter:{RESET}")
    result = word_counter.invoke({"text": "Hello World"})
    print(f"  Result: {result}\n")


async def multiple_structured_outputs():
    """Demonstrate multiple structured output types."""
    print(f"\n{CYAN}{'='*60}")
    print("Multiple Structured Output Types")
    print(f"{'='*60}{RESET}\n")
    
    llm = ChatOllama(model=OLLAMA_SLM, temperature=0)
    
    # Example 1: Research Plan
    print(f"{YELLOW}Example 1: Research Plan{RESET}")
    structured_llm = llm.with_structured_output(ResearchPlan)
    
    response = await structured_llm.ainvoke(
        "Create a research plan for studying the effects of artificial intelligence on job markets"
    )
    
    print(f"{GREEN}Research Plan:{RESET}")
    print(f"  Topic: {response.topic}")
    print(f"  Objectives:")
    for obj in response.objectives:
        print(f"    - {obj}")
    print(f"  Methodology: {response.methodology}")
    print(f"  Timeline: {response.timeline}")
    print()
    
    # Example 2: Code Analysis
    print(f"{YELLOW}Example 2: Code Analysis{RESET}")
    structured_llm = llm.with_structured_output(CodeAnalysis)
    
    code_snippet = """
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    """
    
    response = await structured_llm.ainvoke(
        f"Analyze this code:\n{code_snippet}"
    )
    
    print(f"{GREEN}Code Analysis:{RESET}")
    print(f"  Language: {response.language}")
    print(f"  Complexity: {response.complexity}")
    print(f"  Estimated Lines: {response.estimated_lines}")
    print(f"  Suggestions:")
    for suggestion in response.suggestions:
        print(f"    - {suggestion}")
    print()


async def tool_chaining_example():
    """Demonstrate chaining multiple tools together."""
    print(f"\n{CYAN}{'='*60}")
    print("Tool Chaining Example")
    print(f"{'='*60}{RESET}\n")
    
    print(f"{YELLOW}Task: Calculate 15 + 27, then count words in the result{RESET}\n")
    
    # Step 1: Use calculator
    print(f"{GREEN}Step 1: Calculate 15 + 27{RESET}")
    calc_result = calculator.invoke({"operation": "add", "a": 15, "b": 27})
    print(f"  Result: {calc_result}\n")
    
    # Step 2: Convert to text and count
    print(f"{GREEN}Step 2: Count words in 'The answer is {calc_result}'{RESET}")
    text = f"The answer is {calc_result}"
    count_result = word_counter.invoke({"text": text})
    print(f"  Analysis: {count_result}\n")
    
    # Step 3: Generate related list
    print(f"{GREEN}Step 3: Generate 3 numbers-related items{RESET}")
    # This would work if we had a numbers category
    print(f"  (Would use list_generator here with appropriate category)\n")


async def main():
    """Run all examples."""
    print(f"{GREEN}{'='*60}")
    print("LangChain Tools and Structured Outputs - Examples")
    print(f"{'='*60}{RESET}")
    
    # Run examples
    await manual_tool_usage()
    await tool_calling_example()
    await llm_guided_tool_usage()
    await structured_output_example()
    await multiple_structured_outputs()
    await tool_chaining_example()
    
    print(f"{GREEN}{'='*60}")
    print("All examples completed!")
    print(f"{'='*60}{RESET}\n")
    
    print(f"{YELLOW}Note:{RESET} Tool calling support depends on the Ollama model used.")
    print(f"Some models may not fully support automatic tool calling.")
    print(f"Manual tool usage always works regardless of model capabilities.\n")


if __name__ == "__main__":
    asyncio.run(main())
