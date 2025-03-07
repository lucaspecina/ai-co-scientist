"""Supervisor agent for coordinating all other specialized agents."""

import asyncio
from typing import Any, Dict, List, Optional, Type

from .base_agent import BaseAgent

class SupervisorAgent(BaseAgent):
    """Supervisor agent that coordinates the execution of specialized agents.
    
    This agent manages the task queue, assigns tasks to specialized agents,
    and allocates resources for the AI Co-Scientist system.
    """
    
    def __init__(
        self, 
        model_config: Dict[str, Any], 
        context_memory: Optional[Dict[str, Any]] = None,
        agents: Optional[Dict[str, BaseAgent]] = None
    ):
        """Initialize the supervisor agent.
        
        Args:
            model_config: Configuration for the Gemini model
            context_memory: Shared memory to store agent state and results
            agents: Dictionary of specialized agents to coordinate
        """
        super().__init__(model_config, context_memory)
        self.agents = agents or {}
        self.task_queue = asyncio.Queue()
        self.workers = []
        self.running = False
    
    def register_agent(self, agent_name: str, agent: BaseAgent) -> None:
        """Register a specialized agent with the supervisor.
        
        Args:
            agent_name: Name of the agent
            agent: Agent instance
        """
        self.agents[agent_name] = agent
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the research plan by coordinating specialized agents.
        
        Args:
            task: Task parameters including the research plan configuration
            
        Returns:
            Results from executing the research plan
        """
        research_plan = task.get("research_plan", {})
        self.update_context_memory("research_plan", research_plan)
        
        # Parse the research goal and create a research plan configuration
        await self._parse_research_goal(task.get("research_goal", ""))
        
        # Start worker tasks
        self.running = True
        self.workers = [
            asyncio.create_task(self._worker()) 
            for _ in range(task.get("num_workers", 5))
        ]
        
        # Initialize the task queue with starting tasks
        await self._initialize_task_queue(research_plan)
        
        # Wait for research plan to complete
        max_iterations = task.get("max_iterations", 10)
        for i in range(max_iterations):
            await asyncio.sleep(2.0)  # Wait for worker progress
            
            # Calculate and store statistics
            stats = self._calculate_statistics()
            self.update_context_memory(f"stats_iteration_{i}", stats)
            
            # Check if terminal state is reached
            if self._check_terminal_state(stats, i, max_iterations):
                break
                
            # Update task queue based on current state
            await self._update_task_queue(stats)
        
        # Stop all workers
        self.running = False
        for worker in self.workers:
            worker.cancel()
        
        # Generate final research overview
        if "meta_review" in self.agents:
            research_overview = await self.agents["meta_review"].execute({
                "task_type": "generate_research_overview",
                "top_hypotheses": self.get_from_context_memory("top_hypotheses", [])
            })
            self.update_context_memory("research_overview", research_overview)
        
        return {
            "status": "completed",
            "research_overview": self.get_from_context_memory("research_overview", {}),
            "top_hypotheses": self.get_from_context_memory("top_hypotheses", []),
            "statistics": self.get_from_context_memory(f"stats_iteration_{i}", {})
        }
    
    async def _worker(self) -> None:
        """Worker process that executes tasks from the queue."""
        while self.running:
            try:
                task = await self.task_queue.get()
                agent_name = task.get("agent")
                
                if agent_name in self.agents:
                    agent = self.agents[agent_name]
                    result = await agent.execute(task)
                    
                    # Store results in context memory
                    for key, value in result.items():
                        self.update_context_memory(key, value)
                
                self.task_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in worker: {e}")
                self.task_queue.task_done()
    
    async def _parse_research_goal(self, research_goal: str) -> None:
        """Parse the research goal to derive a research plan configuration.
        
        Args:
            research_goal: Natural language research goal
        """
        if not research_goal:
            return
            
        prompt = f"""
        Parse the following research goal to derive a research plan configuration:
        
        {research_goal}
        
        The configuration should include:
        1. Research objectives
        2. Constraints and preferences
        3. Evaluation criteria (novelty, correctness, testability, etc.)
        4. Any domain-specific knowledge or requirements
        """
        
        response = await self._call_model(prompt)
        research_plan_config = {"raw_goal": research_goal, "parsed_config": response}
        self.update_context_memory("research_plan_config", research_plan_config)
    
    async def _initialize_task_queue(self, research_plan: Dict[str, Any]) -> None:
        """Initialize the task queue with starting tasks.
        
        Args:
            research_plan: Research plan configuration
        """
        # Add initial generation tasks
        if "generation" in self.agents:
            await self.task_queue.put({
                "agent": "generation",
                "task_type": "initial_generation",
                "research_plan": research_plan
            })
    
    async def _update_task_queue(self, stats: Dict[str, Any]) -> None:
        """Update the task queue based on current statistics.
        
        Args:
            stats: Current system statistics
        """
        # Add more generation tasks if needed
        if stats.get("num_hypotheses", 0) < stats.get("target_hypotheses", 20):
            await self.task_queue.put({
                "agent": "generation",
                "task_type": "generate_hypotheses",
                "count": 5
            })
        
        # Add review tasks for unreviewed hypotheses
        unreviewed = stats.get("unreviewed_hypotheses", [])
        for hypothesis_id in unreviewed[:5]:  # Process up to 5 at a time
            await self.task_queue.put({
                "agent": "reflection",
                "task_type": "review_hypothesis",
                "hypothesis_id": hypothesis_id
            })
        
        # Add tournament tasks
        if stats.get("tournament_progress", 0) < 0.8:  # 80% complete
            await self.task_queue.put({
                "agent": "ranking",
                "task_type": "run_tournament_matches",
                "count": 10
            })
        
        # Add evolution tasks for top hypotheses
        top_hypotheses = stats.get("top_hypotheses", [])
        for hypothesis_id in top_hypotheses[:3]:  # Top 3 hypotheses
            await self.task_queue.put({
                "agent": "evolution",
                "task_type": "evolve_hypothesis",
                "hypothesis_id": hypothesis_id
            })
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate system statistics from the context memory.
        
        Returns:
            Dictionary of statistics
        """
        hypotheses = self.get_from_context_memory("hypotheses", [])
        reviewed = self.get_from_context_memory("reviewed_hypotheses", [])
        tournament = self.get_from_context_memory("tournament_state", {})
        
        return {
            "num_hypotheses": len(hypotheses),
            "num_reviewed": len(reviewed),
            "unreviewed_hypotheses": [h["id"] for h in hypotheses if h["id"] not in reviewed],
            "tournament_progress": tournament.get("progress", 0),
            "top_hypotheses": tournament.get("top_ranked", [])[:10],
            "generation_methods": self._count_generation_methods(hypotheses)
        }
    
    def _count_generation_methods(self, hypotheses: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count hypotheses by generation method.
        
        Args:
            hypotheses: List of hypotheses
            
        Returns:
            Dictionary with counts by method
        """
        methods = {}
        for h in hypotheses:
            method = h.get("generation_method", "unknown")
            methods[method] = methods.get(method, 0) + 1
        return methods
    
    def _check_terminal_state(self, stats: Dict[str, Any], iteration: int, max_iterations: int) -> bool:
        """Check if the terminal state for computation has been reached.
        
        Args:
            stats: Current system statistics
            iteration: Current iteration number
            max_iterations: Maximum allowed iterations
            
        Returns:
            True if terminal state is reached, False otherwise
        """
        # Terminal conditions:
        # 1. Maximum iterations reached
        if iteration >= max_iterations - 1:
            return True
            
        # 2. Enough high-quality hypotheses generated and reviewed
        min_hypotheses = 10
        if (stats.get("num_hypotheses", 0) >= min_hypotheses and 
            stats.get("num_reviewed", 0) >= min_hypotheses and
            len(stats.get("top_hypotheses", [])) >= 5):
            
            # Check if tournament is nearly complete
            if stats.get("tournament_progress", 0) > 0.9:  # 90% complete
                return True
        
        return False