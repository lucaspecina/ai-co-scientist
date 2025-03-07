"""Proximity agent for calculating similarity between hypotheses."""

from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent

class ProximityAgent(BaseAgent):
    """Agent for calculating similarity between research hypotheses.
    
    This agent calculates the similarity between hypotheses and builds
    a proximity graph, which helps organize tournament matches and
    showcase diverse ideas.
    """
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a proximity task.
        
        Args:
            task: Task parameters including task type
            
        Returns:
            Proximity calculation results
        """
        task_type = task.get("task_type", "calculate_proximity")
        
        if task_type == "calculate_proximity":
            return await self._calculate_proximity()
        else:
            return {"error": f"Unknown task type: {task_type}"}
    
    async def _calculate_proximity(self) -> Dict[str, Any]:
        """Calculate proximity between all hypotheses.
        
        Returns:
            Proximity graph
        """
        # Get all hypotheses
        hypotheses = self.get_from_context_memory("hypotheses", [])
        
        if len(hypotheses) < 2:
            return {"error": "Not enough hypotheses for proximity calculation"}
        
        # In a real implementation, we would use embeddings to calculate similarity
        # For now, create a simple dummy graph
        proximity_graph = {}
        
        for i, h1 in enumerate(hypotheses):
            proximity_graph[h1["id"]] = []
            
            for j, h2 in enumerate(hypotheses):
                if i != j:
                    # Simple proximity score (would be based on embeddings in real implementation)
                    proximity = 0.5  # Dummy value
                    
                    proximity_graph[h1["id"]].append({
                        "hypothesis_id": h2["id"],
                        "similarity": proximity
                    })
        
        # Store in context memory
        self.update_context_memory("proximity_graph", proximity_graph)
        
        return {"proximity_graph": proximity_graph}