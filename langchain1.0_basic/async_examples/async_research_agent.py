"""
Enhanced Async LangChain Research Agent

This demonstrates advanced async patterns for research:
- Concurrent research queries
- Parallel document analysis
- Streaming responses
- Batch processing with rate limiting

Run:
    python async_examples/async_research_agent.py
"""

import sys
from pathlib import Path
import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# ANSI colors
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
MAGENTA = '\033[95m'
RESET = '\033[0m'

logging.basicConfig(
    format=f'{CYAN}[%(asctime)s]{RESET} {GREEN}%(levelname)s{RESET} %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)

OLLAMA_SLM = "qwen3" # Default Ollama model

class AsyncResearchAgent:
    """
    Enhanced async research agent with concurrent execution capabilities.
    """
    
    def __init__(self, model_name: str = OLLAMA_SLM, temperature: float = 0.7):
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.model_name = model_name
        logging.info(f"Initialized async researcher with {model_name}")
    
    async def research_query(self, query: str) -> Dict[str, Any]:
        """
        Execute a single research query asynchronously.
        
        Args:
            query: Research question
            
        Returns:
            Dict containing research results and metadata
        """
        start_time = datetime.now()
        
        messages = [
            SystemMessage(content="""You are a research assistant. Provide:
            1. Key findings
            2. Supporting evidence
            3. Relevant context
            4. Actionable insights
            
            Be thorough but concise."""),
            HumanMessage(content=query)
        ]
        
        logging.info(f"Researching: {query[:50]}...")
        response = await self.llm.ainvoke(messages)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            "query": query,
            "findings": response.content,
            "model": self.model_name,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }
    
    async def research_concurrent(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Execute multiple research queries concurrently.
        
        Args:
            queries: List of research questions
            
        Returns:
            List of research results
        """
        logging.info(f"{MAGENTA}Starting concurrent research on {len(queries)} queries{RESET}")
        start_time = datetime.now()
        
        # Create tasks for all queries
        tasks = [self.research_query(query) for query in queries]
        
        # Execute all concurrently
        results = await asyncio.gather(*tasks)
        
        total_duration = (datetime.now() - start_time).total_seconds()
        logging.info(f"{GREEN}✓ Completed {len(results)} queries in {total_duration:.2f}s (concurrent){RESET}")
        
        return results
    
    async def research_sequential(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Execute queries sequentially for comparison.
        
        Args:
            queries: List of research questions
            
        Returns:
            List of research results
        """
        logging.info(f"{YELLOW}Starting sequential research on {len(queries)} queries{RESET}")
        start_time = datetime.now()
        
        results = []
        for query in queries:
            result = await self.research_query(query)
            results.append(result)
        
        total_duration = (datetime.now() - start_time).total_seconds()
        logging.info(f"{YELLOW}✓ Completed {len(results)} queries in {total_duration:.2f}s (sequential){RESET}")
        
        return results
    
    async def stream_research(self, query: str):
        """
        Stream research results token by token.
        
        Args:
            query: Research question
        """
        messages = [
            SystemMessage(content="You are a research assistant. Be concise and factual."),
            HumanMessage(content=query)
        ]
        
        logging.info("Streaming research results...")
        print(f"\n{CYAN}Research Findings (streaming):{RESET}")
        print(f"{YELLOW}Query: {query}{RESET}\n")
        
        async for chunk in self.llm.astream(messages):
            print(chunk.content, end="", flush=True)
        
        print("\n")
    
    async def multi_perspective_research(
        self,
        query: str,
        perspectives: List[str]
    ) -> Dict[str, Any]:
        """
        Research from multiple perspectives concurrently.
        
        Args:
            query: Main research question
            perspectives: List of perspectives to analyze from
            
        Returns:
            Dict mapping perspectives to findings
        """
        logging.info(f"{MAGENTA}Researching from {len(perspectives)} perspectives{RESET}")
        
        async def research_from_perspective(perspective: str):
            modified_query = f"From a {perspective} perspective: {query}"
            return await self.research_query(modified_query)
        
        tasks = [research_from_perspective(p) for p in perspectives]
        results = await asyncio.gather(*tasks)
        
        return {perspectives[i]: results[i] for i in range(len(perspectives))}
    
    async def batch_with_rate_limit(
        self,
        queries: List[str],
        max_concurrent: int = 3,
        delay: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Process queries in batches with rate limiting.
        
        Useful when you need to respect API limits or resource constraints.
        
        Args:
            queries: List of research questions
            max_concurrent: Maximum concurrent requests
            delay: Delay between batches in seconds
            
        Returns:
            List of research results
        """
        logging.info(f"{MAGENTA}Processing {len(queries)} queries with rate limit (max {max_concurrent} concurrent){RESET}")
        
        results = []
        
        # Process in batches
        for i in range(0, len(queries), max_concurrent):
            batch = queries[i:i + max_concurrent]
            batch_num = i // max_concurrent + 1
            total_batches = (len(queries) + max_concurrent - 1) // max_concurrent
            
            logging.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} queries)")
            
            # Execute batch concurrently
            tasks = [self.research_query(q) for q in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            # Delay before next batch (except for last batch)
            if i + max_concurrent < len(queries):
                await asyncio.sleep(delay)
        
        return results


async def demo_concurrent_vs_sequential():
    """Demonstrate performance difference between concurrent and sequential."""
    print(f"\n{GREEN}{'='*70}")
    print("DEMO: Concurrent vs Sequential Research")
    print(f"{'='*70}{RESET}\n")
    
    researcher = AsyncResearchAgent()
    
    queries = [
        "What are the latest trends in artificial intelligence?",
        "How does blockchain technology work?",
        "What is quantum computing and its applications?",
        "Explain the principles of cloud-native architecture",
        "What are best practices for API design?"
    ]
    
    print(f"{CYAN}Researching {len(queries)} queries...{RESET}\n")
    
    # Sequential
    print(f"{YELLOW}Method 1: Sequential{RESET}")
    seq_results = await researcher.research_sequential(queries)
    seq_time = sum(r["duration"] for r in seq_results)
    
    await asyncio.sleep(1)
    
    # Concurrent
    print(f"\n{MAGENTA}Method 2: Concurrent{RESET}")
    conc_results = await researcher.research_concurrent(queries)
    conc_time = max(r["duration"] for r in conc_results)
    
    # Results
    speedup = seq_time / conc_time if conc_time > 0 else 0
    print(f"\n{GREEN}{'='*70}")
    print(f"RESULTS:")
    print(f"  Sequential: {seq_time:.2f}s total")
    print(f"  Concurrent: {conc_time:.2f}s total")
    print(f"  Speedup: {speedup:.2f}x faster!")
    print(f"{'='*70}{RESET}\n")


async def demo_streaming():
    """Demonstrate streaming research."""
    print(f"\n{GREEN}{'='*70}")
    print("DEMO: Streaming Research Results")
    print(f"{'='*70}{RESET}\n")
    
    researcher = AsyncResearchAgent()
    query = "What are the key benefits and challenges of microservices architecture?"
    
    await researcher.stream_research(query)


async def demo_multi_perspective():
    """Demonstrate multi-perspective research."""
    print(f"\n{GREEN}{'='*70}")
    print("DEMO: Multi-Perspective Research")
    print(f"{'='*70}{RESET}\n")
    
    researcher = AsyncResearchAgent()
    query = "Implementing AI in healthcare"
    perspectives = ["technical", "ethical", "business", "patient"]
    
    print(f"{CYAN}Researching: {query}{RESET}")
    print(f"{CYAN}Perspectives: {', '.join(perspectives)}{RESET}\n")
    
    results = await researcher.multi_perspective_research(query, perspectives)
    
    for perspective, result in results.items():
        print(f"\n{GREEN}{'='*70}")
        print(f"{perspective.upper()} PERSPECTIVE")
        print(f"{'='*70}{RESET}")
        print(f"{result['findings'][:300]}...")
        print(f"\nDuration: {result['duration']:.2f}s")


async def demo_rate_limited_batch():
    """Demonstrate rate-limited batch processing."""
    print(f"\n{GREEN}{'='*70}")
    print("DEMO: Rate-Limited Batch Processing")
    print(f"{'='*70}{RESET}\n")
    
    researcher = AsyncResearchAgent()
    
    queries = [
        f"Research question #{i+1} about technology trends"
        for i in range(8)
    ]
    
    print(f"{CYAN}Processing {len(queries)} queries with rate limit...{RESET}\n")
    
    results = await researcher.batch_with_rate_limit(
        queries,
        max_concurrent=3,
        delay=0.5
    )
    
    print(f"\n{GREEN}✓ Processed {len(results)} queries with rate limiting{RESET}")


async def demo_progress_tracking():
    """Demonstrate progress tracking with as_completed."""
    print(f"\n{GREEN}{'='*70}")
    print("DEMO: Research with Progress Tracking")
    print(f"{'='*70}{RESET}\n")
    
    researcher = AsyncResearchAgent()
    
    queries = [
        "Machine learning fundamentals",
        "Deep learning architectures",
        "Natural language processing",
        "Computer vision techniques",
        "Reinforcement learning basics"
    ]
    
    print(f"{CYAN}Researching {len(queries)} topics with progress tracking...{RESET}\n")
    
    tasks = [researcher.research_query(q) for q in queries]
    
    for i, task in enumerate(asyncio.as_completed(tasks), 1):
        result = await task
        print(f"{GREEN}✓ {i}/{len(queries)}: {result['query'][:40]}...{RESET} ({result['duration']:.2f}s)")
    
    print(f"\n{GREEN}All research completed!{RESET}\n")


async def main():
    """Run all async research demos."""
    print(f"\n{GREEN}{'='*70}")
    print("Enhanced Async LangChain Research Agent")
    print("Demonstrating Advanced Async Patterns")
    print(f"{'='*70}{RESET}\n")
    
    try:
        # Demo 1: Concurrent vs Sequential
        await demo_concurrent_vs_sequential()
        
        # Demo 2: Streaming
        await demo_streaming()
        
        # Demo 3: Multi-perspective
        await demo_multi_perspective()
        
        # Demo 4: Rate-limited batch
        await demo_rate_limited_batch()
        
        # Demo 5: Progress tracking
        await demo_progress_tracking()
        
        print(f"\n{GREEN}All demos completed!{RESET}")
        
    except Exception as e:
        logging.error(f"Error in demo: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
