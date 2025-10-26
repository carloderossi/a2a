"""
Enhanced Async LangChain Planner Agent

This demonstrates advanced async patterns with LangChain 1.0:
- Structured outputs with Pydantic models
- LCEL (LangChain Expression Language) composition
- Concurrent plan generation
- Batch processing
- Streaming responses
- Parallel multi-model inference
- Proper error handling

Run:
    python async_examples/async_planner_agent.py
"""

import sys
from pathlib import Path
import asyncio
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

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


class ActionPlan(BaseModel):
    """Structured plan output using Pydantic."""
    objective: str = Field(description="Main objective")
    steps: List[str] = Field(description="Actionable steps (5-7 items)")
    timeline: str = Field(description="Timeline estimate")
    resources: List[str] = Field(description="Required resources")
    criteria: List[str] = Field(description="Success criteria")


class PlanResult(BaseModel):
    """Result wrapper with metadata."""
    plan: ActionPlan
    model: str
    duration: float
    timestamp: str
    error: Optional[str] = None


class AsyncPlannerAgent:
    """
    Enhanced async planner with LangChain 1.0 best practices.
    
    Features:
    - Structured output with Pydantic models
    - LCEL composition with | operator
    - Comprehensive error handling
    - Concurrent execution support
    """
    
    def __init__(self, model_name: str = "llama3.1", temperature: float = 0.3):
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.model_name = model_name
        
        # Create prompt template using LCEL
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert strategic planner. Your task is to:
            1. Analyze the research findings thoroughly
            2. Create a clear, actionable plan
            3. Break down complex goals into specific steps
            4. Consider resources and timelines realistically
            5. Define measurable success criteria
            
            Be practical, specific, and comprehensive."""),
            ("human", """Based on this research:

{research}

Create a comprehensive action plan with:
1. Clear objective
2. 5-7 actionable steps
3. Timeline estimate
4. Required resources
5. Success criteria""")
        ])
        
        # Create structured LLM using LCEL composition
        self.structured_llm = self.llm.with_structured_output(ActionPlan)
        
        # Create the complete chain using | operator
        self.chain = self.prompt | self.structured_llm
        
        logging.info(f"Initialized async planner with {model_name} using LCEL")
    
    async def generate_plan(
        self, 
        research: str,
        return_metadata: bool = True
    ) -> Dict[str, Any] | ActionPlan:
        """
        Generate a single plan asynchronously with structured output.
        
        Args:
            research: Research findings to base the plan on
            return_metadata: If True, return dict with metadata; if False, return ActionPlan
            
        Returns:
            Dict with plan and metadata, or ActionPlan object
        """
        start_time = datetime.now()
        error_msg = None
        
        try:
            logging.info(f"Generating plan using LCEL chain...")
            
            # Use LCEL chain for execution
            plan: ActionPlan = await self.chain.ainvoke({"research": research})
            
            duration = (datetime.now() - start_time).total_seconds()
            logging.info(f"{GREEN}✓ Plan generated in {duration:.2f}s{RESET}")
            
            if return_metadata:
                return {
                    "plan": plan,
                    "model": self.model_name,
                    "duration": duration,
                    "timestamp": datetime.now().isoformat(),
                    "error": None
                }
            else:
                return plan
                
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            logging.error(f"{RED}✗ Plan generation failed: {error_msg}{RESET}")
            
            if return_metadata:
                return {
                    "plan": None,
                    "model": self.model_name,
                    "duration": duration,
                    "timestamp": datetime.now().isoformat(),
                    "error": error_msg
                }
            else:
                raise
    
    async def generate_plans_concurrent(
        self, 
        research_items: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple plans concurrently with error handling.
        
        This demonstrates the power of async - all plans are generated
        in parallel rather than sequentially.
        
        Args:
            research_items: List of research findings
            
        Returns:
            List of generated plans with metadata (includes errors)
        """
        logging.info(f"{MAGENTA}Starting concurrent generation of {len(research_items)} plans{RESET}")
        start_time = datetime.now()
        
        try:
            # Create tasks for all plans
            tasks = [self.generate_plan(research) for research in research_items]
            
            # Execute all concurrently with error handling
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.error(f"{RED}✗ Plan {i+1} failed: {result}{RESET}")
                    processed_results.append({
                        "plan": None,
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
                f"{GREEN}✓ Generated {success_count}/{len(results)} plans in {total_duration:.2f}s (concurrent){RESET}"
            )
            
            return processed_results
            
        except Exception as e:
            logging.error(f"{RED}✗ Concurrent generation failed: {e}{RESET}")
            raise
    
    async def generate_plans_sequential(
        self, 
        research_items: List[str]
    ) -> List[Dict[str, Any]]:
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
        for i, research in enumerate(research_items, 1):
            try:
                result = await self.generate_plan(research)
                results.append(result)
            except Exception as e:
                logging.error(f"{RED}✗ Plan {i} failed: {e}{RESET}")
                results.append({
                    "plan": None,
                    "model": self.model_name,
                    "duration": 0.0,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                })
        
        total_duration = (datetime.now() - start_time).total_seconds()
        success_count = sum(1 for r in results if r.get("error") is None)
        
        logging.info(
            f"{YELLOW}✓ Generated {success_count}/{len(results)} plans in {total_duration:.2f}s (sequential){RESET}"
        )
        
        return results
    
    async def stream_plan(self, research: str):
        """
        Stream plan generation token by token.
        
        Note: Streaming doesn't work with structured output,
        so we use regular LLM for this demo.
        
        Args:
            research: Research findings
        """
        try:
            logging.info("Streaming plan generation...")
            print(f"\n{CYAN}Plan (streaming):{RESET}")
            
            # Use prompt without structured output for streaming
            messages = await self.prompt.ainvoke({"research": research[:200]})
            
            async for chunk in self.llm.astream(messages):
                print(chunk.content, end="", flush=True)
            
            print("\n")
            
        except Exception as e:
            logging.error(f"{RED}✗ Streaming failed: {e}{RESET}")
            print(f"\n{RED}Error: {e}{RESET}\n")
    
    async def multi_model_planning(
        self, 
        research: str, 
        models: List[str]
    ) -> Dict[str, Any]:
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
        
        try:
            # Create separate agents for each model
            agents = [AsyncPlannerAgent(model, 0.3) for model in models]
            
            # Generate plans concurrently
            tasks = [agent.generate_plan(research) for agent in agents]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Map results to model names
            output = {}
            for i, (model, result) in enumerate(zip(models, results)):
                if isinstance(result, Exception):
                    logging.error(f"{RED}✗ Model {model} failed: {result}{RESET}")
                    output[model] = {"error": str(result)}
                else:
                    output[model] = result
            
            return output
            
        except Exception as e:
            logging.error(f"{RED}✗ Multi-model planning failed: {e}{RESET}")
            raise


def format_plan(plan: ActionPlan) -> str:
    """Format an ActionPlan for display."""
    output = f"\n{GREEN}{'='*70}\n"
    output += f"OBJECTIVE: {plan.objective}\n"
    output += f"{'='*70}{RESET}\n\n"
    
    output += f"{CYAN}TIMELINE:{RESET} {plan.timeline}\n\n"
    
    output += f"{CYAN}STEPS:{RESET}\n"
    for i, step in enumerate(plan.steps, 1):
        output += f"  {i}. {step}\n"
    
    output += f"\n{CYAN}RESOURCES NEEDED:{RESET}\n"
    for resource in plan.resources:
        output += f"  • {resource}\n"
    
    output += f"\n{CYAN}SUCCESS CRITERIA:{RESET}\n"
    for criterion in plan.criteria:
        output += f"  • {criterion}\n"
    
    return output


async def demo_structured_output():
    """Demonstrate structured output with LCEL."""
    print(f"\n{GREEN}{'='*70}")
    print("DEMO: Structured Output with LCEL")
    print(f"{'='*70}{RESET}\n")
    
    planner = AsyncPlannerAgent()
    research = "AI adoption is growing 40% annually in healthcare"
    
    print(f"{CYAN}Research:{RESET} {research}\n")
    
    try:
        result = await planner.generate_plan(research, return_metadata=False)
        print(format_plan(result))
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")


async def demo_concurrent_vs_sequential():
    """Demonstrate the performance benefit of concurrent execution."""
    print(f"\n{GREEN}{'='*70}")
    print("DEMO: Concurrent vs Sequential Plan Generation")
    print(f"{'='*70}{RESET}\n")
    
    planner = AsyncPlannerAgent()
    
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
    seq_time = sum(r["duration"] for r in seq_results if r.get("error") is None)
    
    await asyncio.sleep(1)
    
    # Concurrent execution
    print(f"\n{MAGENTA}Method 2: Concurrent (all at once){RESET}")
    conc_results = await planner.generate_plans_concurrent(research_items)
    conc_time = max(r["duration"] for r in conc_results if r.get("error") is None)
    
    # Show speedup
    speedup = seq_time / conc_time if conc_time > 0 else 0
    print(f"\n{GREEN}{'='*70}")
    print(f"RESULTS:")
    print(f"  Sequential: {seq_time:.2f}s total")
    print(f"  Concurrent: {conc_time:.2f}s total")
    print(f"  Speedup: {speedup:.2f}x faster!")
    print(f"{'='*70}{RESET}\n")


async def demo_error_handling():
    """Demonstrate robust error handling."""
    print(f"\n{GREEN}{'='*70}")
    print("DEMO: Error Handling in Async Operations")
    print(f"{'='*70}{RESET}\n")
    
    planner = AsyncPlannerAgent()
    
    # Mix of valid and invalid inputs
    research_items = [
        "Valid research about AI",
        "",  # Empty - might cause error
        "Another valid research topic",
        None,  # Invalid - will cause error
        "Final valid research"
    ]
    
    print(f"{CYAN}Processing {len(research_items)} items (some will fail)...{RESET}\n")
    
    results = await planner.generate_plans_concurrent(research_items)
    
    print(f"\n{CYAN}Results:{RESET}")
    for i, result in enumerate(results, 1):
        if result.get("error"):
            print(f"  {RED}✗ Item {i}: {result['error']}{RESET}")
        else:
            print(f"  {GREEN}✓ Item {i}: Success ({result['duration']:.2f}s){RESET}")
    
    success_count = sum(1 for r in results if r.get("error") is None)
    print(f"\n{GREEN}Successfully processed {success_count}/{len(results)} items{RESET}\n")


async def demo_streaming():
    """Demonstrate streaming plan generation."""
    print(f"\n{GREEN}{'='*70}")
    print("DEMO: Streaming Plan Generation")
    print(f"{'='*70}{RESET}\n")
    
    planner = AsyncPlannerAgent()
    research = "Implementing a microservices architecture for a monolithic e-commerce platform"
    
    await planner.stream_plan(research)


async def demo_batch_processing():
    """Demonstrate batch processing with progress tracking."""
    print(f"\n{GREEN}{'='*70}")
    print("DEMO: Batch Processing with Progress")
    print(f"{'='*70}{RESET}\n")
    
    planner = AsyncPlannerAgent()
    
    research_batch = [
        f"Research finding #{i+1}: Technology trend analysis"
        for i in range(10)
    ]
    
    print(f"{CYAN}Processing batch of {len(research_batch)} items...{RESET}\n")
    
    # Process with progress tracking
    tasks = [planner.generate_plan(research) for research in research_batch]
    
    for i, task in enumerate(asyncio.as_completed(tasks), 1):
        try:
            result = await task
            if result.get("error"):
                print(f"{RED}✗ Completed {i}/{len(research_batch)}{RESET} (error: {result['error']})")
            else:
                print(f"{GREEN}✓ Completed {i}/{len(research_batch)}{RESET} (took {result['duration']:.2f}s)")
        except Exception as e:
            print(f"{RED}✗ Completed {i}/{len(research_batch)}{RESET} (exception: {e})")
    
    print(f"\n{GREEN}Batch processing complete!{RESET}\n")


async def main():
    """Run all async demos."""
    print(f"\n{GREEN}{'='*70}")
    print("Enhanced Async LangChain Planner Agent")
    print("LangChain 1.0 Best Practices")
    print(f"{'='*70}{RESET}\n")
    
    try:
        # Demo 1: Structured output with LCEL
        await demo_structured_output()
        
        # Demo 2: Concurrent vs Sequential
        await demo_concurrent_vs_sequential()
        
        # Demo 3: Error handling
        await demo_error_handling()
        
        # Demo 4: Streaming
        await demo_streaming()
        
        # Demo 5: Batch processing
        await demo_batch_processing()
        
        print(f"{GREEN}All demos completed!{RESET}")
        
    except Exception as e:
        logging.error(f"Error in demo: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
