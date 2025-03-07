"""Base agent class for the AI Co-Scientist system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseAgent(ABC):
    """Base class for all agents in the AI Co-Scientist system.
    
    This class defines the common interface for all specialized agents.
    Each agent has access to the Gemini model and can perform specific tasks.
    """
    
    def __init__(self, model_config: Dict[str, Any], context_memory: Optional[Dict[str, Any]] = None):
        """Initialize the agent.
        
        Args:
            model_config: Configuration for the Gemini model
            context_memory: Shared memory to store agent state and results
        """
        self.model_config = model_config
        self.context_memory = context_memory or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task assigned to this agent.
        
        Args:
            task: Task parameters and context
            
        Returns:
            Task results and updated context
        """
        pass
    
    async def _call_model(self, prompt: str, **kwargs) -> str:
        """Call the Gemini model with the given prompt.
        
        Args:
            prompt: Instruction prompt for the model
            **kwargs: Additional parameters for the model call
            
        Returns:
            Model response as a string
        """
        # This would be implemented to call the actual Gemini API
        # For now, we'll return a placeholder
        return f"Model response to: {prompt[:30]}..."
    
    def update_context_memory(self, key: str, value: Any) -> None:
        """Update the shared context memory.
        
        Args:
            key: Memory key
            value: Value to store
        """
        if self.context_memory is not None:
            self.context_memory[key] = value
    
    def get_from_context_memory(self, key: str, default: Any = None) -> Any:
        """Get a value from the shared context memory.
        
        Args:
            key: Memory key
            default: Default value if key doesn't exist
            
        Returns:
            Stored value or default
        """
        if self.context_memory is None:
            return default
        return self.context_memory.get(key, default)