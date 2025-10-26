"""
Enhanced Async LangChain Research Agent

This demonstrates advanced async patterns with LangChain 1.0:
- Structured outputs with Pydantic models
- LCEL composition with | operator
- Concurrent research queries
- Parallel document analysis
- Streaming responses
- Batch processing with rate limiting
- Comprehensive error handling

Run:
    python async_examples/async_research_agent.py
"""

import sys
from pathlib import Path
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# ANSI colors
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
MAGENTA = '\033[95m'
RED = '\033[91m'
RESET = '\033[0m'

logging.basicConfig(
    format=f'{CYAN}[%(asctime)s]{RESET} {GREEN}%(levelname)s{RESET} %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)


class ResearchFindings(BaseModel):
    """Structured research output using Pydantic."""
    topic: str = Field(description="Research topic")
    key_findings: List[str] = Field(description="Key findings (3-5 items)")
    supporting_evidence: List[str] = Field(description="Supporting evidence")
    implications: str = Field(description="Practical implications")
    confidence: str = Field(description="Confidence level: high, medium, or low")


class AsyncResearchAgent:
    """
    Enhanced async research agent with LangChain 1.0 best practices.
    
    Features:
    - Structured output with Pydantic models
    - LCEL composition with | operator
    - Comprehensive error handling
    - Concurrent execution support
    """
    
    def __init__(self, model_name: str = "llama3.1", temperature: float = 0.7):
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.model_name = model_name
        
        # Create prompt template using LCEL
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research assistant. Provide:
            1. Key findings
            2. Supporting evidence
            3. Relevant context
            4. Actionable insights
            
            Be thorough but concise."""),
            ("human", "{query}")
        ])
        
        # Create the complete chain using | operator
        self.chain = self.prompt | self.llm
        
        logging.info(f"Initialized async researcher with {model_name} using LCEL")
    
    async def research_query(
        self, 
        query: str,
        structured: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a single research query asynchronously.
        
        Args:
            query: Research question
            structured: If True, use structured output
            
        Returns:
            Dict containing research results and metadata
        """
        start_time = datetime.now()
        error_msg = None
        
        try:
            logging.info(f"Researching: {query[:50]}...")
            
            if structured:
                # Use structured output chain
                structured_llm = self.llm.with_structured_output(ResearchFindings)
                structured_chain = self.prompt | structured_llm
                findings = await structured_chain.ainvoke({"query": query})
                result_content = findings
            else:
                # Use regular chain
                response = await self.chain.ainvoke({"query": query})
                result_content = response.content
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return {
                "query": query,
                "findings": result_content,
                "model": self.model_name,
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
                "error": None
            }
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            logging.error(f"{RED}✗ Research failed: {error_msg}{RESET}")
            
            return {
                "query": query,
                "findings": None,
                "model": self.model_name,
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
                "error": error_msg
            }
    
    async def research_concurrent(
        self, 
        queries: List[str],
        structured: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple research queries concurrently with error handling.
        
        Args:
            queries: List of research questions
            structured: If True, use structured output
            
        Returns:
            List of research results (includes errors)
        """
        logging.info(f"{MAGENTA}Starting concurrent research on {len(queries)} queries{RESET}")
        start_time = datetime.now()
        
        try:
            # Create tasks for all queries
            tasks = [self.research_query(query, structured) for query in queries]
            
            # Execute all concurrently with error handling
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.error(f"{RED}✗ Query {i+1} failed: {result}{RESET}")
                    processed_results.append({
                        "query": queries[i] if i < len(queries) else "unknown",
                        "findings": None,
                        "model": self.model_name,
                        "duration": 0.0,
                        "timestamp": datetime.now().isoformat(),
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)
            
            total_duration = (datetime.now() - start_time).total_seconds()
            success_count = sum(1 for r in processed_results if r.get("error") is None)
            
            logging.info(
                f"{GREEN}✓ Completed {success_count}/{len(results)} queries in {total_duration:.2f}s (concurrent){RESET}"
            )
            
            return processed_results
            
        except Exception as e:
            logging.error(f"{RED}✗ Concurrent research failed: {e}{RESET}")
            raise
    
    async def research_sequential(
        self, 
        queries: List[str]
    ) -> List[Dict[str, Any]]:
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
        for i, query in enumerate(queries, 1):
            try:
                result = await self.research_query(query)
                results.append(result)
            except Exception as e:
                logging.error(f"{RED}✗ Query {i} failed: {e}{RESET}")
                results.append({
                    "query": query,
                    "findings": None,
                    "model": self.model_name,
                    "duration": 0.0,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                })
        
        total_duration = (datetime.now() - start_time).total_seconds()
        success_count = sum(1 for r in results if r.get("error") is None)
        
        logging.info(
            f"{YELLOW}✓ Completed {success_count}/{len(results)} queries in {total_duration:.2f}s (sequential){RESET}"
        )
        
        return results
    
    async def stream_research(self, query: str):
        """
        Stream research results token by token.
        
        Args:
            query: Research question
        """
        try:
            logging.info("Streaming research results...")
            print(f"\n{CYAN}Research Findings (streaming):{RESET}")
            print(f"{YELLOW}Query: {query}{RESET}\n")
            
            messages = await self.prompt.ainvoke({"query": query})
            
            async for chunk in self.llm.astream(messages):
                print(chunk.content, end="", flush=True)
            
            print("\n")
            
        except Exception as e:
            logging.error(f"{RED}✗ Streaming failed: {e}{RESET}")
            print(f"\n{RED}Error: {e}{RESET}\n")
    
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
        
        try:
            async def research_from_perspective(perspective: str):
                modified_query = f"From a {perspective} perspective: {query}"
                return await self.research_query(modified_query)
            
            tasks = [research_from_perspective(p) for p in perspectives]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            output = {}
            for i, (perspective, result) in enumerate(zip(perspectives, results)):
                if isinstance(result, Exception):
                    logging.error(f"{RED}✗ Perspective {perspective} failed: {result}{RESET}")
                    output[perspective] = {"error": str(result)}
                else:
                    output[perspective] = result
            
            return output
            
        except Exception as e:
            logging.error(f"{RED}✗ Multi-perspective research failed: {e}{RESET}")
            raise
    
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
        logging.info(
            f"{MAGENTA}Processing {len(queries)} queries with rate limit "
            f"(max {max_concurrent} concurrent){RESET}"
        )
        
        try:
            results = []
            
            # Process in batches
            for i in range(0, len(queries), max_concurrent):
                batch = queries[i:i + max_concurrent]
                batch_num = i // max_concurrent + 1
                total_batches = (len(queries) + max_concurrent - 1) // max_concurrent
                
                logging.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} queries)")
                
                # Execute batch concurrently
                tasks = [self.research_query(q) for q in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle exceptions in batch
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        results.append({
                            "query": batch[j],
                            "findings": None,
                            "model": self.model_name,
                            "duration": 0.0,
                            "timestamp": datetime.now().isoformat(),
                            "error": str(result)
                        })
                    else:
                        results.append(result)
                
                # Delay before next batch (except for last batch)
                if i + max_concurrent < len(queries):
                    await asyncio.sleep(delay)
            
            return results
            
        except Exception as e:
            logging.error(f"{RED}✗ Batch processing failed: {e}{RESET}")
            raise


async def demo_structured_output():
    """Demonstrate structured output."""
    print(f"\n{GREEN}{'='*70}")
    print("DEMO: Structured Research Output")
    print(f"{'='*70}{RESET}\n")
    
    researcher = AsyncResearchAgent()
    query = "What are the latest trends in artificial intelligence?"
    
    print(f"{CYAN}Query:{RESET} {query}\n")
    
    result = await researcher.research_query(query, structured=True)
    
    if result.get("error"):
        print(f"{RED}Error: {result['error']}{RESET}")
    else:
        findings: ResearchFindings = result["findings"]
        print(f"{GREEN}Topic:{RESET} {findings.topic}")
        print(f"{GREEN}Confidence:{RESET} {findings.confidence}\n")
        print(f"{CYAN}Key Findings:{RESET}")
        for i, finding in enumerate(findings.key_findings, 1):
            print(f"  {i}. {finding}")
        print(f"\n{CYAN}Implications:{RESET} {findings.implications}\n")


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
    seq_time = sum(r["duration"] for r in seq_results if r.get("error") is None)
    
    await asyncio.sleep(1)
    
    # Concurrent
    print(f"\n{MAGENTA}Method 2: Concurrent{RESET}")
    conc_results = await researcher.research_concurrent(queries)
    conc_time = max(r["duration"] for r in conc_results if r.get("error") is None)
    
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
        
        if isinstance(result, dict) and result.get("error"):
            print(f"{RED}Error: {result['error']}{RESET}")
        elif isinstance(result, dict):
            findings = result.get("findings", "")
            if isinstance(findings, str):
                print(f"{findings[:300]}...")
            print(f"\nDuration: {result.get('duration', 0):.2f}s")


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
    
    success_count = sum(1 for r in results if r.get("error") is None)
    print(f"\n{GREEN}✓ Processed {success_count}/{len(results)} queries with rate limiting{RESET}\n")


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
        try:
            result = await task
            if result.get("error"):
                print(f"{RED}✗ {i}/{len(queries)}: Error{RESET}")
            else:
                query_short = result['query'][:40]
                print(f"{GREEN}✓ {i}/{len(queries)}: {query_short}...{RESET} ({result['duration']:.2f}s)")
        except Exception as e:
            print(f"{RED}✗ {i}/{len(queries)}: Exception - {e}{RESET}")
    
    print(f"\n{GREEN}All research completed!{RESET}\n")


async def main():
    """Run all async research demos."""
    print(f"\n{GREEN}{'='*70}")
    print("Enhanced Async LangChain Research Agent")
    print("LangChain 1.0 Best Practices")
    print(f"{'='*70}{RESET}\n")
    
    try:
        # Demo 1: Structured output
        await demo_structured_output()
        
        # Demo 2: Concurrent vs Sequential
        await demo_concurrent_vs_sequential()
        
        # Demo 3: Streaming
        await demo_streaming()
        
        # Demo 4: Multi-perspective
        await demo_multi_perspective()
        
        # Demo 5: Rate-limited batch
        await demo_rate_limited_batch()
        
        # Demo 6: Progress tracking
        await demo_progress_tracking()
        
        print(f"\n{GREEN}All demos completed!{RESET}")
        
    except Exception as e:
        logging.error(f"Error in demo: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
