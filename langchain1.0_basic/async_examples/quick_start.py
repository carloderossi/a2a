"""
Quick Start: Async LangChain Basics

This is your starting point for understanding async with LangChain.
Run this first to see the core concepts in action.

Run:
    python async_examples/quick_start.py
"""

import sys
from pathlib import Path
import asyncio
import time

sys.path.append(str(Path(__file__).parent.parent))

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# ANSI colors
GREEN = '\033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
RESET = '\033[0m'

OLLAMA_SLM = "qwen3" # Default Ollama model

# ============================================================================
# Example 1: Sync vs Async - The Basic Difference
# ============================================================================

async def example1():
    """Basic sync vs async comparison."""
    print(f"\n{GREEN}{'='*70}")
    print("Quick Start: Async LangChain Basics")
    print(f"{'='*70}{RESET}\n")
    
    print(f"{CYAN}Example 1: Basic Sync vs Async{RESET}\n")

    llm = ChatOllama(model=OLLAMA_SLM, temperature=0.7)

    # Synchronous way (blocking)
    def sync_call():
        """Traditional synchronous call - blocks until complete."""
        # messages = [HumanMessage(content="Say hello in one word")]
        messages = [HumanMessage(content="What's your name?")]
        response = llm.invoke(messages)
        return response.content

    # Asynchronous way (non-blocking)
    async def async_call():
        """Asynchronous call - allows other work while waiting."""
        # messages = [HumanMessage(content="Say hello in one word")]
        messages = [HumanMessage(content="What's your name?")]
        response = await llm.ainvoke(messages)
        return response.content

    print("Sync call:")
    start = time.time()
    result = sync_call()
    print(f"  Result: {result}")
    print(f"  Time: {time.time() - start:.2f}s\n")

    print("Async call:")
    start = time.time()
    result = await async_call()
    print(f"  Result: {result}")
    print(f"  Time: {time.time() - start:.2f}s\n")

    print(f"{YELLOW}💡 Same result, same time - async shines with multiple calls!{RESET}\n")


# ============================================================================
# Example 2: The Real Power - Concurrent Calls
# ============================================================================

async def example2():
    """Sequential vs concurrent comparison."""
    print(f"{CYAN}Example 2: Sequential vs Concurrent (The Magic!){RESET}\n")

    llm = ChatOllama(model=OLLAMA_SLM, temperature=0.7)
    
    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?"
    ]

    async def sequential_calls():
        """Process one at a time - slow."""
        start = time.time()
        results = []
        
        for question in questions:
            messages = [HumanMessage(content=f"{question} (one word)")]
            response = await llm.ainvoke(messages)
            results.append(response.content)
        
        duration = time.time() - start
        return results, duration

    async def concurrent_calls():
        """Process all at once - fast!"""
        start = time.time()
        
        # Create all tasks
        tasks = []
        for question in questions:
            messages = [HumanMessage(content=f"{question} (one word)")]
            tasks.append(llm.ainvoke(messages))
        
        # Execute all at once
        responses = await asyncio.gather(*tasks)
        results = [r.content for r in responses]
        
        duration = time.time() - start
        return results, duration

    # Run sequential
    print("Sequential (one at a time):")
    seq_results, seq_time = await sequential_calls()
    for q, r in zip(questions, seq_results):
        print(f"  {q} → {r}")
    print(f"  ⏱️  Total: {seq_time:.2f}s\n")

    # Run concurrent
    print("Concurrent (all at once):")
    conc_results, conc_time = await concurrent_calls()
    for q, r in zip(questions, conc_results):
        print(f"  {q} → {r}")
    print(f"  ⏱️  Total: {conc_time:.2f}s\n")

    speedup = seq_time / conc_time if conc_time > 0 else 1
    print(f"{GREEN}🚀 Speedup: {speedup:.2f}x faster!{RESET}\n")


# ============================================================================
# Example 3: Streaming Responses
# ============================================================================

async def example3():
    """Streaming demonstration."""
    print(f"{CYAN}Example 3: Streaming (Token by Token){RESET}\n")

    llm = ChatOllama(model=OLLAMA_SLM, temperature=0.7)
    
    messages = [
        SystemMessage(content="Be concise."),
        HumanMessage(content="List 3 programming languages")
    ]
    
    print("Streaming response: \n", end="", flush=True)
    async for chunk in llm.astream(messages):
        print(chunk.content, end="", flush=True)
    print("\n")


# ============================================================================
# Example 4: Progress Tracking with as_completed
# ============================================================================

async def example4():
    """Progress tracking demonstration."""
    print(f"{CYAN}Example 4: Progress Tracking{RESET}\n")

    llm = ChatOllama(model=OLLAMA_SLM, temperature=0.7)
    
    tasks = []
    for i in range(1, 6):
        messages = [HumanMessage(content=f"Say '{i}' in one word")]
        tasks.append(llm.ainvoke(messages))
    
    print("Processing 5 queries with progress:\n")
    
    # Process and show as each completes
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        result = await coro
        print(f"  ✓ {i}/5 completed: {result.content}")
    print()


# ============================================================================
# Example 5: Error Handling
# ============================================================================

async def example5():
    """Error handling demonstration."""
    print(f"{CYAN}Example 5: Graceful Error Handling{RESET}\n")

    llm = ChatOllama(model=OLLAMA_SLM, temperature=0.7)
    
    async def task_that_might_fail(n, should_fail=False):
        if should_fail:
            raise ValueError(f"Task {n} failed!")
        messages = [HumanMessage(content=f"Say 'success {n}'")]
        response = await llm.ainvoke(messages)
        return response.content
    
    tasks = [
        task_that_might_fail(1, False),
        task_that_might_fail(2, True),   # This will fail
        task_that_might_fail(3, False),
        task_that_might_fail(4, True),   # This will fail
        task_that_might_fail(5, False),
    ]
    
    # return_exceptions=True prevents one failure from stopping all
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    print("Results with error handling (2 and 4 should fail):\n")
    for i, result in enumerate(results, 1):
        if isinstance(result, Exception):
            print(f"  ❌ Task {i}: {result}")
        else:
            print(f"  ✅ Task {i}: {result}")
    print()


# ============================================================================
# Summary
# ============================================================================

def print_summary():
    """Print the summary."""
    print(f"{GREEN}{'='*70}")
    print("Summary: When to Use Async")
    print(f"{'='*70}{RESET}\n")

    print(f"{GREEN}✅ Use Async When:{RESET}")
    print("   • Making multiple LLM calls")
    print("   • Building web APIs")
    print("   • Processing batches")
    print("   • Need better performance")
    print()

    print(f"{YELLOW}⚠️  Stick with Sync When:{RESET}")
    print("   • Simple single queries")
    print("   • Jupyter notebooks")
    print("   • Quick scripts")
    print("   • No concurrency needed")
    print()

    print(f"{CYAN}🎯 Key Takeaways:{RESET}")
    print("   1. Use await llm.ainvoke() instead of llm.invoke()")
    print("   2. Use asyncio.gather() for concurrent execution")
    print("   3. Use as_completed() for progress tracking")
    print("   4. Use return_exceptions=True for error handling")
    print()

    print(f"{GREEN}Next Steps:{RESET}")
    print("   • Try async_planner_agent.py for more patterns")
    print("   • Run performance_comparison.py for detailed metrics")
    print("   • Check concurrent_agents_demo.py for multi-agent systems")
    print()

    print(f"{GREEN}Happy async coding! 🚀{RESET}\n")


async def main():
    """Run all examples in a single event loop."""
    try:
        await example1()
        await example2()
        await example3()
        await example4()
        await example5()
        print_summary()
    except Exception as e:
        print(f"\n{YELLOW}Error: {e}{RESET}")
        print(f"{YELLOW}Make sure Ollama is running with llama3.1 model!{RESET}\n")


if __name__ == "__main__":
    # Use a single event loop for all examples
    asyncio.run(main())
