"""
Enhanced Async LangChain Planner Agent

This demonstrates advanced async patterns with LangChain:
- Concurrent plan generation
- Batch processing
- Streaming responses
- Parallel multi-model inference

Run:
    python async_examples/async_planner_agent.py
"""

import sys
from pathlib import Path
import asyncio
import uuid
import logging
from typing import List, Dict, Any
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field
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

class ActionPlan(BaseModel):
    """Structured plan output."""
    objective: str = Field(description="Main objective")
    steps: List[str] = Field(description="Actionable steps")
    timeline: str = Field(description="Timeline estimate")
    resources: List[str] = Field(description="Required resources")
    criteria: List[str] = Field(description="Success criteria")


class AsyncPlannerAgent:
    """
    Enhanced async planner with concurrent execution capabilities.
    """
    
    def __init__(self, model_name: str = OLLAMA_SLM, temperature: float = 0.3):
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.model_name = model_name
        logging.info(f"Initialized async planner with {model_name}")
    
    async def generate_plan(self, research: str) -> Dict[str, Any]:
        """
        Generate a single plan asynchronously.
        
        Args:
            research: Research findings to base the plan on
            
        Returns:
            Dict containing plan and metadata
        """
        start_time = datetime.now()
        
        prompt = f"""Based on this research:

{research}

Create a comprehensive action plan with:
1. Clear objective
2. 5-7 actionable steps
3. Timeline estimate
4. Required resources
5. Success criteria"""

        messages = [
            SystemMessage(content="You are an expert strategic planner. Create clear, actionable plans."),
            HumanMessage(content=prompt)
        ]
        
        logging.info(f"Generating plan...")
        response = await self.llm.ainvoke(messages)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            "plan": response.content,
            "model": self.model_name,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }
    
    async def generate_plans_concurrent(self, research_items: List[str]) -> List[Dict[str, Any]]:
        """
        Generate multiple plans concurrently.
        
        This demonstrates the power of async - all plans are generated
        in parallel rather than sequentially.
        
        Args:
            research_items: List of research findings
            
        Returns:
            List of generated plans with metadata
        """
        logging.info(f"{MAGENTA}Starting concurrent generation of {len(research_items)} plans{RESET}")
        start_time = datetime.now()
        
        # Create tasks for all plans
        tasks = [self.generate_plan(research) for research in research_items]
        
        # Execute all concurrently
        results = await asyncio.gather(*tasks)
        
        total_duration = (datetime.now() - start_time).total_seconds()
        logging.info(f"{GREEN}✓ Generated {len(results)} plans in {total_duration:.2f}s (concurrent){RESET}")
        
        return results
    
    async def generate_plans_sequential(self, research_items: List[str]) -> List[Dict[str, Any]]:
        """
        Generate plans sequentially for comparison.
        
        Args:
            research_items: List of research findings
            
        Returns:
            List of generated plans with metadata
        """
        logging.info(f"{YELLOW}Starting sequential generation of {len(research_items)} plans{RESET}")
        start_time = datetime.now()
        
        results = []
        for research in research_items:
            result = await self.generate_plan(research)
            results.append(result)
        
        total_duration = (datetime.now() - start_time).total_seconds()
        logging.info(f"{YELLOW}✓ Generated {len(results)} plans in {total_duration:.2f}s (sequential){RESET}")
        
        return results
    
    async def stream_plan(self, research: str):
        """
        Stream plan generation token by token.
        
        Args:
            research: Research findings
        """
        prompt = f"Create a plan based on: {research[:200]}..."
        
        messages = [
            SystemMessage(content="You are a strategic planner."),
            HumanMessage(content=prompt)
        ]
        
        logging.info("Streaming plan generation...")
        print(f"\n{CYAN}Plan (streaming):{RESET}")
        
        async for chunk in self.llm.astream(messages):
            print(chunk.content, end="", flush=True)
        
        print("\n")
    
    async def multi_model_planning(self, research: str, models: List[str]) -> Dict[str, Any]:
        """
        Generate plans using multiple models concurrently.
        
        This shows how async enables easy comparison of different models.
        
        Args:
            research: Research findings
            models: List of model names to use
            
        Returns:
            Dict mapping model names to their plans
        """
        logging.info(f"{MAGENTA}Generating plans with {len(models)} models concurrently{RESET}")
        
        # Create separate agents for each model
        agents = [AsyncPlannerAgent(model, 0.3) for model in models]
        
        # Generate plans concurrently
        tasks = [agent.generate_plan(research) for agent in agents]
        results = await asyncio.gather(*tasks)
        
        # Map results to model names
        return {result["model"]: result for result in results}


async def demo_concurrent_vs_sequential():
    """Demonstrate the performance benefit of concurrent execution."""
    print(f"\n{GREEN}{'='*70}")
    print("DEMO: Concurrent vs Sequential Plan Generation")
    print(f"{'='*70}{RESET}\n")
    
    planner = AsyncPlannerAgent()
    
    # Sample research items
    research_items = [
        "AI adoption is growing 40% annually in healthcare",
        "Quantum computing shows promise for drug discovery",
        "Edge computing reduces latency by 60% in IoT applications",
        "Blockchain improves supply chain transparency",
        "5G enables new augmented reality applications"
    ]
    
    print(f"{CYAN}Generating {len(research_items)} plans...{RESET}\n")
    
    # Sequential execution
    print(f"{YELLOW}Method 1: Sequential (one at a time){RESET}")
    seq_results = await planner.generate_plans_sequential(research_items)
    seq_time = sum(r["duration"] for r in seq_results)
    
    await asyncio.sleep(1)  # Brief pause between methods
    
    # Concurrent execution
    print(f"\n{MAGENTA}Method 2: Concurrent (all at once){RESET}")
    conc_results = await planner.generate_plans_concurrent(research_items)
    conc_time = max(r["duration"] for r in conc_results)
    
    # Show speedup
    speedup = seq_time / conc_time if conc_time > 0 else 0
    print(f"\n{GREEN}{'='*70}")
    print(f"RESULTS:")
    print(f"  Sequential: {seq_time:.2f}s total")
    print(f"  Concurrent: {conc_time:.2f}s total")
    print(f"  Speedup: {speedup:.2f}x faster!")
    print(f"{'='*70}{RESET}\n")


async def demo_streaming():
    """Demonstrate streaming plan generation."""
    print(f"\n{GREEN}{'='*70}")
    print("DEMO: Streaming Plan Generation")
    print(f"{'='*70}{RESET}\n")
    
    planner = AsyncPlannerAgent()
    research = "Implementing a microservices architecture for a monolithic e-commerce platform"
    
    await planner.stream_plan(research)


async def demo_multi_model():
    """Demonstrate concurrent multi-model planning."""
    print(f"\n{GREEN}{'='*70}")
    print("DEMO: Multi-Model Concurrent Planning")
    print(f"{'='*70}{RESET}\n")
    
    research = "Developing an AI-powered customer service system"
    models = [OLLAMA_SLM, "llama3.2"]  # Add more models if available
    
    planner = AsyncPlannerAgent()
    results = await planner.multi_model_planning(research, models)
    
    print(f"{CYAN}Plans generated by {len(results)} models:{RESET}\n")
    for model, result in results.items():
        print(f"{GREEN}Model: {model}{RESET}")
        print(f"Duration: {result['duration']:.2f}s")
        print(f"Plan preview: {result['plan'][:200]}...")
        print()


async def demo_batch_processing():
    """Demonstrate batch processing with progress tracking."""
    print(f"\n{GREEN}{'='*70}")
    print("DEMO: Batch Processing with Progress")
    print(f"{'='*70}{RESET}\n")
    
    planner = AsyncPlannerAgent()
    
    # Larger batch
    research_batch = [
        f"Research finding #{i+1}: Technology trend analysis"
        for i in range(10)
    ]
    
    print(f"{CYAN}Processing batch of {len(research_batch)} items...{RESET}\n")
    
    # Process with progress tracking
    tasks = [planner.generate_plan(research) for research in research_batch]
    
    for i, task in enumerate(asyncio.as_completed(tasks), 1):
        result = await task
        print(f"{GREEN}✓ Completed {i}/{len(research_batch)}{RESET} (took {result['duration']:.2f}s)")
    
    print(f"\n{GREEN}Batch processing complete!{RESET}\n")


async def main():
    """Run all async demos."""
    print(f"\n{GREEN}{'='*70}")
    print("Enhanced Async LangChain Planner Agent")
    print("Demonstrating Advanced Async Patterns")
    print(f"{'='*70}{RESET}\n")
    
    try:
        # Demo 1: Concurrent vs Sequential
        await demo_concurrent_vs_sequential()
        
        # Demo 2: Streaming
        await demo_streaming()
        
        # Demo 3: Multi-model (if multiple models available)
        # await demo_multi_model()
        
        # Demo 4: Batch processing
        await demo_batch_processing()
        
        print(f"{GREEN}All demos completed!{RESET}")
        
    except Exception as e:
        logging.error(f"Error in demo: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
