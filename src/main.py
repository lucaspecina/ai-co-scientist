"""Main entry point for the AI Co-Scientist system."""

import asyncio
import argparse
import json
from typing import Dict, Any, Optional

from agents import (
    BaseAgent,
    SupervisorAgent,
    GenerationAgent,
    ReflectionAgent,
    RankingAgent,
    ProximityAgent,
    EvolutionAgent,
    MetaReviewAgent,
)

async def run_co_scientist(
    research_goal: str,
    output_file: Optional[str] = None,
    max_iterations: int = 10,
    num_workers: int = 5,
    model_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run the AI Co-Scientist system on a research goal.
    
    Args:
        research_goal: Natural language research goal
        output_file: Path to save results (optional)
        max_iterations: Maximum number of iterations to run
        num_workers: Number of worker processes
        model_config: Configuration for the Gemini model
        
    Returns:
        Results from the co-scientist system
    """
    # Initialize shared context memory
    context_memory = {}
    
    # Create model configuration if not provided
    if model_config is None:
        model_config = {
            "model_name": "gemini-2.0",
            "temperature": 0.7,
            "max_tokens": 8192
        }
    
    # Create agents
    supervisor = SupervisorAgent(model_config, context_memory)
    
    # Register specialized agents
    supervisor.register_agent("generation", GenerationAgent(model_config, context_memory))
    supervisor.register_agent("reflection", ReflectionAgent(model_config, context_memory))
    supervisor.register_agent("ranking", RankingAgent(model_config, context_memory))
    supervisor.register_agent("proximity", ProximityAgent(model_config, context_memory))
    supervisor.register_agent("evolution", EvolutionAgent(model_config, context_memory))
    supervisor.register_agent("meta_review", MetaReviewAgent(model_config, context_memory))
    
    # Execute the research plan
    results = await supervisor.execute({
        "research_goal": research_goal,
        "max_iterations": max_iterations,
        "num_workers": num_workers
    })
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="AI Co-Scientist system")
    parser.add_argument("--goal", type=str, required=True, help="Research goal as a string or file path")
    parser.add_argument("--output", type=str, help="Path to save results")
    parser.add_argument("--iterations", type=int, default=10, help="Maximum number of iterations")
    parser.add_argument("--workers", type=int, default=5, help="Number of worker processes")
    parser.add_argument("--model", type=str, default="gemini-2.0", help="Model name to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Model temperature")
    
    args = parser.parse_args()
    
    # Load research goal from file if it's a file path
    if args.goal.endswith('.txt') or args.goal.endswith('.md'):
        try:
            with open(args.goal, 'r') as f:
                research_goal = f.read()
        except FileNotFoundError:
            research_goal = args.goal
    else:
        research_goal = args.goal
    
    # Create model configuration
    model_config = {
        "model_name": args.model,
        "temperature": args.temperature,
        "max_tokens": 8192
    }
    
    # Run the co-scientist system
    asyncio.run(run_co_scientist(
        research_goal=research_goal,
        output_file=args.output,
        max_iterations=args.iterations,
        num_workers=args.workers,
        model_config=model_config
    ))

if __name__ == "__main__":
    main()