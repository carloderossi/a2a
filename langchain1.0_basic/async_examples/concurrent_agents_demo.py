"""
Concurrent Agent Coordination Demo

This demonstrates:
- Running multiple agents concurrently
- Research → Planning pipeline with async
- Parallel multi-query processing
- Agent coordination patterns

Run:
    python async_examples/concurrent_agents_demo.py
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
BLUE = '\033[94m'
RESET = '\033[0m'

logging.basicConfig(
    format=f'{CYAN}[%(asctime)s]{RESET} {GREEN}%(levelname)s{RESET} %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)


class ResearchAgent:
    """Simplified async research agent."""
    
    def __init__(self, name: str, model: str = "llama3.1"):
        self.name = name
        self.llm = ChatOllama(model=model, temperature=0.7)
        logging.info(f"Initialized {name}")
    
    async def research(self, query: str) -> str:
        """Execute research query."""
        messages = [
            SystemMessage(content="You are a research assistant. Be concise and factual."),
            HumanMessage(content=query)
        ]
        
        logging.info(f"{BLUE}[{self.name}]{RESET} Researching: {query[:50]}...")
        response = await self.llm.ainvoke(messages)
        logging.info(f"{GREEN}[{self.name}]{RESET} ✓ Research complete")
        
        return response.content


class PlannerAgent:
    """Simplified async planner agent."""
    
    def __init__(self, name: str, model: str = "llama3.1"):
        self.name = name
        self.llm = ChatOllama(model=model, temperature=0.3)
        logging.info(f"Initialized {name}")
    
    async def plan(self, research: str) -> str:
        """Create plan from research."""
        prompt = f"""Based on this research:

{research}

Create a 5-step action plan."""

        messages = [
            SystemMessage(content="You are a strategic planner. Be clear and actionable."),
            HumanMessage(content=prompt)
        ]
        
        logging.info(f"{MAGENTA}[{self.name}]{RESET} Planning based on research...")
        response = await self.llm.ainvoke(messages)
        logging.info(f"{GREEN}[{self.name}]{RESET} ✓ Plan complete")
        
        return response.content


async def demo_sequential_pipeline():
    """
    Traditional sequential approach: Research → Plan
    Each step waits for the previous one.
    """
    print(f"\n{YELLOW}{'='*70}")
    print("DEMO 1: Sequential Pipeline (Research → Plan)")
    print(f"{'='*70}{RESET}\n")
    
    researcher = ResearchAgent("Researcher-1")
    planner = PlannerAgent("Planner-1")
    
    query = "Latest trends in edge computing"
    
    start = datetime.now()
    
    # Step 1: Research (wait for completion)
    print(f"{CYAN}Step 1: Research{RESET}")
    research = await researcher.research(query)
    research_time = (datetime.now() - start).total_seconds()
    print(f"Research completed in {research_time:.2f}s\n")
    
    # Step 2: Planning (wait for completion)
    print(f"{CYAN}Step 2: Planning{RESET}")
    plan = await planner.plan(research)
    total_time = (datetime.now() - start).total_seconds()
    plan_time = total_time - research_time
    print(f"Planning completed in {plan_time:.2f}s\n")
    
    print(f"{GREEN}Total Sequential Time: {total_time:.2f}s{RESET}\n")


async def demo_concurrent_multiple_queries():
    """
    Process multiple queries concurrently.
    All research happens in parallel, then all planning happens in parallel.
    """
    print(f"\n{MAGENTA}{'='*70}")
    print("DEMO 2: Concurrent Multiple Queries")
    print(f"{'='*70}{RESET}\n")
    
    queries = [
        "AI in healthcare",
        "Blockchain in finance",
        "IoT security",
    ]
    
    researcher = ResearchAgent("Researcher-2")
    planner = PlannerAgent("Planner-2")
    
    start = datetime.now()
    
    # Phase 1: Research all queries concurrently
    print(f"{CYAN}Phase 1: Concurrent Research ({len(queries)} queries){RESET}")
    research_tasks = [researcher.research(q) for q in queries]
    research_results = await asyncio.gather(*research_tasks)
    research_time = (datetime.now() - start).total_seconds()
    print(f"{GREEN}✓ All research completed in {research_time:.2f}s{RESET}\n")
    
    # Phase 2: Plan all results concurrently
    print(f"{CYAN}Phase 2: Concurrent Planning ({len(research_results)} plans){RESET}")
    planning_tasks = [planner.plan(r) for r in research_results]
    plans = await asyncio.gather(*planning_tasks)
    total_time = (datetime.now() - start).total_seconds()
    plan_time = total_time - research_time
    print(f"{GREEN}✓ All planning completed in {plan_time:.2f}s{RESET}\n")
    
    print(f"{GREEN}Total Concurrent Time: {total_time:.2f}s{RESET}\n")
    print(f"{CYAN}Processed {len(queries)} queries in parallel!{RESET}\n")


async def demo_parallel_agents():
    """
    Multiple agents working on different tasks simultaneously.
    """
    print(f"\n{BLUE}{'='*70}")
    print("DEMO 3: Parallel Multi-Agent System")
    print(f"{'='*70}{RESET}\n")
    
    # Create multiple agents
    researcher1 = ResearchAgent("Researcher-A")
    researcher2 = ResearchAgent("Researcher-B")
    planner1 = PlannerAgent("Planner-A")
    planner2 = PlannerAgent("Planner-B")
    
    start = datetime.now()
    
    # All agents work simultaneously on different tasks
    print(f"{CYAN}All agents working in parallel...{RESET}\n")
    
    results = await asyncio.gather(
        researcher1.research("Machine learning trends"),
        researcher2.research("Cloud computing innovations"),
        planner1.plan("Focus on automation and efficiency"),
        planner2.plan("Focus on scalability and performance"),
    )
    
    total_time = (datetime.now() - start).total_seconds()
    
    print(f"\n{GREEN}{'='*70}")
    print(f"4 agents completed work in {total_time:.2f}s (parallel)")
    print(f"If done sequentially, this would take 4x longer!")
    print(f"{'='*70}{RESET}\n")


async def demo_research_planning_pipeline():
    """
    Optimized pipeline: Start planning as soon as any research completes.
    """
    print(f"\n{GREEN}{'='*70}")
    print("DEMO 4: Optimized Research→Planning Pipeline")
    print(f"{'='*70}{RESET}\n")
    
    queries = [
        "Quantum computing applications",
        "5G network benefits",
        "Augmented reality in education",
    ]
    
    researcher = ResearchAgent("Researcher-3")
    planner = PlannerAgent("Planner-3")
    
    print(f"{CYAN}Processing {len(queries)} queries with optimized pipeline{RESET}\n")
    
    start = datetime.now()
    completed = 0
    
    # Start all research tasks
    research_tasks = {
        asyncio.create_task(researcher.research(q)): q 
        for q in queries
    }
    
    plans = []
    
    # As each research completes, immediately start planning
    for coro in asyncio.as_completed(research_tasks.keys()):
        research_result = await coro
        completed += 1
        
        query = research_tasks[coro]
        print(f"{GREEN}✓ Research {completed}/{len(queries)} complete{RESET}")
        print(f"{MAGENTA}  → Starting planning immediately...{RESET}")
        
        # Start planning without waiting for other research
        plan = await planner.plan(research_result)
        plans.append(plan)
        print(f"{GREEN}  ✓ Plan {completed}/{len(queries)} complete{RESET}\n")
    
    total_time = (datetime.now() - start).total_seconds()
    
    print(f"{GREEN}{'='*70}")
    print(f"Optimized pipeline completed in {total_time:.2f}s")
    print(f"Planning started as soon as research was ready!")
    print(f"{'='*70}{RESET}\n")


async def demo_error_handling():
    """
    Demonstrate error handling in concurrent operations.
    """
    print(f"\n{YELLOW}{'='*70}")
    print("DEMO 5: Error Handling in Concurrent Operations")
    print(f"{'='*70}{RESET}\n")
    
    async def task_that_might_fail(task_id: int, fail: bool = False):
        """Simulate a task that might fail."""
        await asyncio.sleep(0.5)
        if fail:
            raise ValueError(f"Task {task_id} failed!")
        return f"Task {task_id} succeeded"
    
    print(f"{CYAN}Running 5 tasks, 2 will fail...{RESET}\n")
    
    tasks = [
        task_that_might_fail(1, fail=False),
        task_that_might_fail(2, fail=True),
        task_that_might_fail(3, fail=False),
        task_that_might_fail(4, fail=True),
        task_that_might_fail(5, fail=False),
    ]
    
    # Method 1: gather with return_exceptions
    print(f"{CYAN}Method 1: gather(return_exceptions=True){RESET}")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(results, 1):
        if isinstance(result, Exception):
            print(f"{YELLOW}  Task {i}: Failed - {result}{RESET}")
        else:
            print(f"{GREEN}  Task {i}: {result}{RESET}")
    
    print(f"\n{GREEN}All tasks handled, failures didn't stop the others!{RESET}\n")


async def main():
    """Run all concurrent demos."""
    print(f"\n{GREEN}{'='*70}")
    print("Concurrent Agent Coordination Demonstrations")
    print("Showing the Power of Async in Multi-Agent Systems")
    print(f"{'='*70}{RESET}\n")
    
    try:
        # Demo 1: Sequential pipeline (baseline)
        await demo_sequential_pipeline()
        
        # Demo 2: Concurrent multiple queries
        await demo_concurrent_multiple_queries()
        
        # Demo 3: Parallel agents
        await demo_parallel_agents()
        
        # Demo 4: Optimized pipeline
        await demo_research_planning_pipeline()
        
        # Demo 5: Error handling
        await demo_error_handling()
        
        print(f"\n{GREEN}{'='*70}")
        print("All Demonstrations Completed!")
        print(f"{'='*70}{RESET}\n")
        
        print(f"{CYAN}Key Takeaways:{RESET}")
        print(f"  • Async enables true parallelism with I/O-bound operations")
        print(f"  • Multiple agents can work simultaneously")
        print(f"  • Significant performance improvements (2-4x typical)")
        print(f"  • Better resource utilization")
        print(f"  • Maintains code clarity and organization\n")
        
    except Exception as e:
        logging.error(f"Error in demo: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
