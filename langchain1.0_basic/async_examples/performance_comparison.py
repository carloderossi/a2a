"""
Performance Comparison: Sync vs Async

This script provides a clear comparison of synchronous vs asynchronous
execution patterns with actual timing metrics.

Run:
    python async_examples/performance_comparison.py
"""

import sys
from pathlib import Path
import asyncio
import time
import logging
from typing import List
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# ANSI colors
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
MAGENTA = '\033[95m'
RED = '\033[91m'
RESET = '\033[0m'

logging.basicConfig(
    format=f'{CYAN}[%(asctime)s]{RESET} %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)


class PerformanceTest:
    """Test harness for comparing sync vs async performance."""
    
    def __init__(self, model: str = "llama3.1"):
        self.model = model
        self.llm = ChatOllama(model=model, temperature=0.5)
    
    # === SYNCHRONOUS METHODS ===
    
    def sync_single_query(self, query: str) -> dict:
        """Execute a single query synchronously."""
        start = time.time()
        
        messages = [
            SystemMessage(content="You are a helpful assistant. Be brief."),
            HumanMessage(content=query)
        ]
        
        # Synchronous invoke
        response = self.llm.invoke(messages)
        
        duration = time.time() - start
        return {
            "query": query,
            "response": response.content,
            "duration": duration
        }
    
    def sync_multiple_queries(self, queries: List[str]) -> List[dict]:
        """Execute multiple queries synchronously (one at a time)."""
        results = []
        for query in queries:
            result = self.sync_single_query(query)
            results.append(result)
        return results
    
    # === ASYNCHRONOUS METHODS ===
    
    async def async_single_query(self, query: str) -> dict:
        """Execute a single query asynchronously."""
        start = time.time()
        
        messages = [
            SystemMessage(content="You are a helpful assistant. Be brief."),
            HumanMessage(content=query)
        ]
        
        # Asynchronous invoke
        response = await self.llm.ainvoke(messages)
        
        duration = time.time() - start
        return {
            "query": query,
            "response": response.content,
            "duration": duration
        }
    
    async def async_multiple_queries_sequential(self, queries: List[str]) -> List[dict]:
        """Execute multiple queries asynchronously but sequentially."""
        results = []
        for query in queries:
            result = await self.async_single_query(query)
            results.append(result)
        return results
    
    async def async_multiple_queries_concurrent(self, queries: List[str]) -> List[dict]:
        """Execute multiple queries concurrently (the async advantage!)."""
        tasks = [self.async_single_query(query) for query in queries]
        results = await asyncio.gather(*tasks)
        return list(results)


def print_results(title: str, results: List[dict], total_time: float):
    """Pretty print results."""
    print(f"\n{GREEN}{title}{RESET}")
    print(f"{CYAN}{'='*70}{RESET}")
    print(f"Queries: {len(results)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average per query: {total_time/len(results):.2f}s")
    print()
    
    for i, result in enumerate(results, 1):
        query = result['query'][:50]
        duration = result['duration']
        print(f"  {i}. {query}... ({duration:.2f}s)")


async def run_sync_comparison():
    """Run synchronous test (but wrap in async for consistency)."""
    print(f"\n{YELLOW}{'='*70}")
    print("TEST 1: SYNCHRONOUS EXECUTION")
    print(f"{'='*70}{RESET}")
    print(f"{YELLOW}Processing queries one at a time...{RESET}\n")
    
    test = PerformanceTest()
    
    queries = [
        "What is Python?",
        "Explain async programming",
        "What is LangChain?",
        "Describe REST APIs",
        "What is Docker?"
    ]
    
    start = time.time()
    results = test.sync_multiple_queries(queries)
    total_time = time.time() - start
    
    print_results("SYNCHRONOUS RESULTS", results, total_time)
    
    return total_time


async def run_async_sequential_comparison():
    """Run async sequential test."""
    print(f"\n{CYAN}{'='*70}")
    print("TEST 2: ASYNC SEQUENTIAL EXECUTION")
    print(f"{'='*70}{RESET}")
    print(f"{CYAN}Using async but still processing one at a time...{RESET}\n")
    
    test = PerformanceTest()
    
    queries = [
        "What is Python?",
        "Explain async programming",
        "What is LangChain?",
        "Describe REST APIs",
        "What is Docker?"
    ]
    
    start = time.time()
    results = await test.async_multiple_queries_sequential(queries)
    total_time = time.time() - start
    
    print_results("ASYNC SEQUENTIAL RESULTS", results, total_time)
    
    return total_time


async def run_async_concurrent_comparison():
    """Run async concurrent test."""
    print(f"\n{MAGENTA}{'='*70}")
    print("TEST 3: ASYNC CONCURRENT EXECUTION")
    print(f"{'='*70}{RESET}")
    print(f"{MAGENTA}Processing all queries simultaneously!{RESET}\n")
    
    test = PerformanceTest()
    
    queries = [
        "What is Python?",
        "Explain async programming",
        "What is LangChain?",
        "Describe REST APIs",
        "What is Docker?"
    ]
    
    start = time.time()
    results = await test.async_multiple_queries_concurrent(queries)
    total_time = time.time() - start
    
    print_results("ASYNC CONCURRENT RESULTS", results, total_time)
    
    return total_time


def print_comparison_summary(sync_time: float, async_seq_time: float, async_conc_time: float):
    """Print comparison summary."""
    print(f"\n{GREEN}{'='*70}")
    print("PERFORMANCE COMPARISON SUMMARY")
    print(f"{'='*70}{RESET}\n")
    
    print(f"{YELLOW}Synchronous:{RESET}          {sync_time:.2f}s")
    print(f"{CYAN}Async Sequential:{RESET}      {async_seq_time:.2f}s")
    print(f"{MAGENTA}Async Concurrent:{RESET}      {async_conc_time:.2f}s")
    
    print(f"\n{GREEN}Speedup Analysis:{RESET}")
    
    async_vs_sync = sync_time / async_seq_time if async_seq_time > 0 else 1
    print(f"  Async Sequential vs Sync: {async_vs_sync:.2f}x")
    
    conc_vs_sync = sync_time / async_conc_time if async_conc_time > 0 else 1
    print(f"  Async Concurrent vs Sync: {conc_vs_sync:.2f}x {GREEN}← BEST!{RESET}")
    
    conc_vs_seq = async_seq_time / async_conc_time if async_conc_time > 0 else 1
    print(f"  Concurrent vs Sequential: {conc_vs_seq:.2f}x")
    
    time_saved = sync_time - async_conc_time
    percent_saved = (time_saved / sync_time * 100) if sync_time > 0 else 0
    
    print(f"\n{GREEN}Time Saved:{RESET} {time_saved:.2f}s ({percent_saved:.1f}% faster)")
    print(f"\n{CYAN}{'='*70}{RESET}\n")
    
    print(f"{GREEN}Key Insights:{RESET}")
    print(f"  • Async Sequential ≈ Sync (minimal overhead)")
    print(f"  • Async Concurrent = REAL speedup!")
    print(f"  • Best for I/O-bound operations (like LLM calls)")
    print(f"  • Speedup scales with number of concurrent operations")
    print()


async def main():
    """Run all performance tests."""
    print(f"\n{GREEN}{'='*70}")
    print("LangChain Performance Comparison: Sync vs Async")
    print(f"{'='*70}{RESET}")
    print(f"\n{CYAN}Testing with 5 queries to Ollama...{RESET}\n")
    
    try:
        # Test 1: Synchronous
        sync_time = await run_sync_comparison()
        
        await asyncio.sleep(1)  # Brief pause
        
        # Test 2: Async Sequential
        async_seq_time = await run_async_sequential_comparison()
        
        await asyncio.sleep(1)  # Brief pause
        
        # Test 3: Async Concurrent
        async_conc_time = await run_async_concurrent_comparison()
        
        # Summary
        print_comparison_summary(sync_time, async_seq_time, async_conc_time)
        
        print(f"{GREEN}Performance testing completed!{RESET}\n")
        
    except Exception as e:
        logging.error(f"Error during performance test: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
