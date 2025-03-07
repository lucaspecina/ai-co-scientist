"""Specialized agents for the AI Co-Scientist system."""

from .base_agent import BaseAgent
from .supervisor_agent import SupervisorAgent
from .generation_agent import GenerationAgent
from .reflection_agent import ReflectionAgent
from .ranking_agent import RankingAgent
from .proximity_agent import ProximityAgent
from .evolution_agent import EvolutionAgent
from .meta_review_agent import MetaReviewAgent

__all__ = [
    "BaseAgent",
    "SupervisorAgent",
    "GenerationAgent",
    "ReflectionAgent",
    "RankingAgent",
    "ProximityAgent", 
    "EvolutionAgent",
    "MetaReviewAgent",
]