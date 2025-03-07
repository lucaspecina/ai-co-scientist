"""Generation agent for creating novel research hypotheses."""

import uuid
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent

class GenerationAgent(BaseAgent):
    """Agent for generating novel research hypotheses and proposals.
    
    This agent uses various techniques including literature exploration,
    simulated scientific debates, iterative assumptions identification,
    and research expansion to generate hypotheses.
    """
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a hypothesis generation task.
        
        Args:
            task: Task parameters including research plan and task type
            
        Returns:
            Generated hypotheses and updated context
        """
        task_type = task.get("task_type", "generate_hypotheses")
        
        if task_type == "initial_generation":
            return await self._initial_generation(task)
        elif task_type == "generate_hypotheses":
            return await self._generate_hypotheses(task)
        elif task_type == "literature_exploration":
            return await self._literature_exploration(task)
        elif task_type == "simulated_debate":
            return await self._simulated_debate(task)
        elif task_type == "assumptions_identification":
            return await self._assumptions_identification(task)
        elif task_type == "research_expansion":
            return await self._research_expansion(task)
        else:
            return {"error": f"Unknown task type: {task_type}"}
    
    async def _initial_generation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate initial focus areas and hypotheses for the research goal.
        
        Args:
            task: Task parameters including research plan
            
        Returns:
            Initial hypotheses and focus areas
        """
        research_plan = task.get("research_plan", {})
        research_goal = research_plan.get("raw_goal", "")
        
        if not research_goal:
            research_goal = self.get_from_context_memory("research_plan_config", {}).get("raw_goal", "")
        
        prompt = f"""
        Based on the following research goal:
        
        {research_goal}
        
        Generate 3-5 initial focus areas for exploration, each with a brief description.
        """
        
        focus_areas_response = await self._call_model(prompt)
        focus_areas = self._parse_focus_areas(focus_areas_response)
        
        # Generate initial hypotheses for each focus area
        hypotheses = []
        for area in focus_areas:
            area_hypotheses = await self._generate_hypotheses_for_focus_area(area, research_goal)
            hypotheses.extend(area_hypotheses)
        
        # Store in context memory
        existing_hypotheses = self.get_from_context_memory("hypotheses", [])
        all_hypotheses = existing_hypotheses + hypotheses
        self.update_context_memory("hypotheses", all_hypotheses)
        self.update_context_memory("focus_areas", focus_areas)
        
        return {
            "focus_areas": focus_areas,
            "generated_hypotheses": hypotheses,
            "hypotheses": all_hypotheses
        }
    
    async def _generate_hypotheses(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a batch of hypotheses using various techniques.
        
        Args:
            task: Task parameters including count and constraints
            
        Returns:
            Generated hypotheses
        """
        count = task.get("count", 3)
        method = task.get("method", "mixed")
        
        hypotheses = []
        
        if method == "literature":
            # Generate hypotheses through literature exploration
            for _ in range(count):
                hypothesis = await self._literature_exploration({})
                if "hypothesis" in hypothesis:
                    hypotheses.append(hypothesis["hypothesis"])
        
        elif method == "debate":
            # Generate hypotheses through simulated debates
            for _ in range(count):
                hypothesis = await self._simulated_debate({})
                if "hypothesis" in hypothesis:
                    hypotheses.append(hypothesis["hypothesis"])
        
        elif method == "assumptions":
            # Generate hypotheses through assumptions identification
            for _ in range(count):
                hypothesis = await self._assumptions_identification({})
                if "hypothesis" in hypothesis:
                    hypotheses.append(hypothesis["hypothesis"])
        
        elif method == "expansion":
            # Generate hypotheses through research expansion
            for _ in range(count):
                hypothesis = await self._research_expansion({})
                if "hypothesis" in hypothesis:
                    hypotheses.append(hypothesis["hypothesis"])
        
        else:  # mixed - use all methods
            methods = ["literature", "debate", "assumptions", "expansion"]
            for i in range(count):
                method_index = i % len(methods)
                method_name = methods[method_index]
                
                if method_name == "literature":
                    hypothesis = await self._literature_exploration({})
                elif method_name == "debate":
                    hypothesis = await self._simulated_debate({})
                elif method_name == "assumptions":
                    hypothesis = await self._assumptions_identification({})
                else:  # expansion
                    hypothesis = await self._research_expansion({})
                
                if "hypothesis" in hypothesis:
                    hypotheses.append(hypothesis["hypothesis"])
        
        # Store in context memory
        existing_hypotheses = self.get_from_context_memory("hypotheses", [])
        all_hypotheses = existing_hypotheses + hypotheses
        self.update_context_memory("hypotheses", all_hypotheses)
        
        return {
            "generated_hypotheses": hypotheses,
            "hypotheses": all_hypotheses
        }
    
    async def _literature_exploration(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a hypothesis through literature exploration.
        
        Args:
            task: Task parameters (optional)
            
        Returns:
            Generated hypothesis
        """
        research_goal = self.get_from_context_memory("research_plan_config", {}).get("raw_goal", "")
        
        # In a real implementation, this would use web search to find relevant articles
        # For now, we'll simulate the process
        
        prompt = f"""
        For the research goal:
        
        {research_goal}
        
        1. Identify 3 relevant search queries that would help explore this topic.
        2. Summarize the key findings from the literature (simulate searching and reading several papers).
        3. Based on these findings, generate a novel research hypothesis that:
           - Builds on existing literature
           - Addresses a gap in current knowledge
           - Is testable and falsifiable
           - Is aligned with the research goal
        
        Format the response as:
        SEARCH QUERIES:
        - Query 1
        - Query 2
        - Query 3
        
        LITERATURE SUMMARY:
        [Summary of key findings from relevant papers]
        
        HYPOTHESIS:
        [Clear statement of the hypothesis]
        
        RATIONALE:
        [Explanation of the hypothesis and how it builds on existing literature]
        
        TESTABILITY:
        [How this hypothesis could be tested experimentally]
        """
        
        response = await self._call_model(prompt)
        
        # Parse the response and create a structured hypothesis
        hypothesis = self._parse_hypothesis_from_literature(response)
        hypothesis["id"] = str(uuid.uuid4())
        hypothesis["generation_method"] = "literature_exploration"
        
        return {"hypothesis": hypothesis}
    
    async def _simulated_debate(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a hypothesis through simulated scientific debate.
        
        Args:
            task: Task parameters (optional)
            
        Returns:
            Generated hypothesis
        """
        research_goal = self.get_from_context_memory("research_plan_config", {}).get("raw_goal", "")
        
        prompt = f"""
        For the research goal:
        
        {research_goal}
        
        Simulate a scientific debate among three experts with different perspectives 
        to generate a novel research hypothesis:
        
        Expert A's initial hypothesis:
        [Generate an initial hypothesis related to the research goal]
        
        Expert B's critique:
        [Critique Expert A's hypothesis, pointing out limitations or alternative interpretations]
        
        Expert C's synthesis:
        [Offer a different perspective that addresses some of Expert B's concerns]
        
        Expert A's response:
        [Defend the initial hypothesis while acknowledging valid critiques]
        
        Expert B's counter:
        [Refine the critique based on Expert A's response]
        
        Expert C's final synthesis:
        [Propose a refined hypothesis that incorporates the best elements of the debate]
        
        FINAL HYPOTHESIS:
        [Clear statement of the final hypothesis after the debate]
        
        RATIONALE:
        [Explanation of the hypothesis and how it emerged from the debate]
        
        TESTABILITY:
        [How this hypothesis could be tested experimentally]
        """
        
        response = await self._call_model(prompt)
        
        # Parse the response and create a structured hypothesis
        hypothesis = self._parse_hypothesis_from_debate(response)
        hypothesis["id"] = str(uuid.uuid4())
        hypothesis["generation_method"] = "simulated_debate"
        
        return {"hypothesis": hypothesis}
    
    async def _assumptions_identification(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a hypothesis through iterative assumptions identification.
        
        Args:
            task: Task parameters (optional)
            
        Returns:
            Generated hypothesis
        """
        research_goal = self.get_from_context_memory("research_plan_config", {}).get("raw_goal", "")
        
        prompt = f"""
        For the research goal:
        
        {research_goal}
        
        Generate a hypothesis through iterative assumptions identification:
        
        1. Identify 3-5 testable assumptions that, if proven true, would contribute to the research goal.
        2. For each assumption, identify 2-3 sub-assumptions or logical steps.
        3. Assess the plausibility of each assumption and sub-assumption.
        4. Combine the most plausible assumptions into a coherent hypothesis.
        
        Format the response as:
        
        ASSUMPTION 1:
        [Statement of assumption]
        - Sub-assumption 1.1: [Statement]
        - Sub-assumption 1.2: [Statement]
        Plausibility assessment: [High/Medium/Low]
        
        ASSUMPTION 2:
        [Statement of assumption]
        - Sub-assumption 2.1: [Statement]
        - Sub-assumption 2.2: [Statement]
        Plausibility assessment: [High/Medium/Low]
        
        [Continue for all assumptions]
        
        COMBINED HYPOTHESIS:
        [Clear statement of the combined hypothesis]
        
        RATIONALE:
        [Explanation of how the assumptions combine into a coherent hypothesis]
        
        TESTABILITY:
        [How this hypothesis could be tested experimentally]
        """
        
        response = await self._call_model(prompt)
        
        # Parse the response and create a structured hypothesis
        hypothesis = self._parse_hypothesis_from_assumptions(response)
        hypothesis["id"] = str(uuid.uuid4())
        hypothesis["generation_method"] = "assumptions_identification"
        
        return {"hypothesis": hypothesis}
    
    async def _research_expansion(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a hypothesis through research expansion.
        
        Args:
            task: Task parameters (optional)
            
        Returns:
            Generated hypothesis
        """
        research_goal = self.get_from_context_memory("research_plan_config", {}).get("raw_goal", "")
        existing_hypotheses = self.get_from_context_memory("hypotheses", [])
        
        # Get summaries of existing hypotheses
        hypothesis_summaries = "\n".join([
            f"- {h.get('title', 'Untitled hypothesis')}: {h.get('summary', 'No summary')}"
            for h in existing_hypotheses[:5]  # Use up to 5 existing hypotheses
        ])
        
        prompt = f"""
        For the research goal:
        
        {research_goal}
        
        Consider the following existing hypotheses:
        
        {hypothesis_summaries}
        
        Generate a novel research hypothesis that:
        1. Explores an area not covered by existing hypotheses
        2. Addresses a different aspect of the research goal
        3. Uses a different approach or perspective
        4. Is testable and falsifiable
        
        Format the response as:
        
        NOVEL DIRECTION:
        [Describe a research direction not covered by existing hypotheses]
        
        HYPOTHESIS:
        [Clear statement of the hypothesis]
        
        RATIONALE:
        [Explanation of the hypothesis and how it differs from existing ones]
        
        TESTABILITY:
        [How this hypothesis could be tested experimentally]
        """
        
        response = await self._call_model(prompt)
        
        # Parse the response and create a structured hypothesis
        hypothesis = self._parse_hypothesis_from_expansion(response)
        hypothesis["id"] = str(uuid.uuid4())
        hypothesis["generation_method"] = "research_expansion"
        
        return {"hypothesis": hypothesis}
    
    async def _generate_hypotheses_for_focus_area(self, focus_area: Dict[str, Any], research_goal: str) -> List[Dict[str, Any]]:
        """Generate hypotheses for a specific focus area.
        
        Args:
            focus_area: Focus area information
            research_goal: Research goal
            
        Returns:
            List of generated hypotheses
        """
        area_title = focus_area.get("title", "")
        area_description = focus_area.get("description", "")
        
        prompt = f"""
        For the research goal:
        
        {research_goal}
        
        Generate 2 novel research hypotheses for the following focus area:
        
        FOCUS AREA: {area_title}
        DESCRIPTION: {area_description}
        
        For each hypothesis, provide:
        1. A clear hypothesis statement
        2. Rationale and background
        3. How it could be tested experimentally
        4. Potential implications if proven true
        
        FORMAT:
        
        HYPOTHESIS 1:
        [Clear statement of the hypothesis]
        
        RATIONALE:
        [Explanation of the hypothesis and its background]
        
        TESTABILITY:
        [How this hypothesis could be tested experimentally]
        
        IMPLICATIONS:
        [Potential implications if proven true]
        
        HYPOTHESIS 2:
        [Clear statement of the hypothesis]
        
        RATIONALE:
        [Explanation of the hypothesis and its background]
        
        TESTABILITY:
        [How this hypothesis could be tested experimentally]
        
        IMPLICATIONS:
        [Potential implications if proven true]
        """
        
        response = await self._call_model(prompt)
        
        # Parse the response and create structured hypotheses
        hypotheses = self._parse_multiple_hypotheses(response)
        
        # Add metadata to each hypothesis
        for hypothesis in hypotheses:
            hypothesis["id"] = str(uuid.uuid4())
            hypothesis["generation_method"] = "focus_area_exploration"
            hypothesis["focus_area"] = area_title
        
        return hypotheses
    
    def _parse_focus_areas(self, response: str) -> List[Dict[str, Any]]:
        """Parse focus areas from model response.
        
        Args:
            response: Model response string
            
        Returns:
            List of focus areas as dictionaries
        """
        # In a real implementation, this would parse the structured response
        # For now, we'll return dummy data
        return [
            {"id": str(uuid.uuid4()), "title": "Focus Area 1", "description": "Description of focus area 1"},
            {"id": str(uuid.uuid4()), "title": "Focus Area 2", "description": "Description of focus area 2"},
            {"id": str(uuid.uuid4()), "title": "Focus Area 3", "description": "Description of focus area 3"},
        ]
    
    def _parse_hypothesis_from_literature(self, response: str) -> Dict[str, Any]:
        """Parse hypothesis from literature exploration response.
        
        Args:
            response: Model response string
            
        Returns:
            Structured hypothesis dictionary
        """
        # In a real implementation, this would parse the structured response
        # For now, we'll return dummy data
        return {
            "title": "Hypothesis from literature",
            "statement": "Statement of hypothesis from literature exploration",
            "rationale": "Rationale for the hypothesis",
            "testability": "How this hypothesis could be tested",
            "literature_summary": "Summary of relevant literature",
            "search_queries": ["query1", "query2", "query3"]
        }
    
    def _parse_hypothesis_from_debate(self, response: str) -> Dict[str, Any]:
        """Parse hypothesis from simulated debate response.
        
        Args:
            response: Model response string
            
        Returns:
            Structured hypothesis dictionary
        """
        # In a real implementation, this would parse the structured response
        # For now, we'll return dummy data
        return {
            "title": "Hypothesis from debate",
            "statement": "Statement of hypothesis from simulated debate",
            "rationale": "Rationale for the hypothesis",
            "testability": "How this hypothesis could be tested",
            "debate_summary": "Summary of the simulated debate"
        }
    
    def _parse_hypothesis_from_assumptions(self, response: str) -> Dict[str, Any]:
        """Parse hypothesis from assumptions identification response.
        
        Args:
            response: Model response string
            
        Returns:
            Structured hypothesis dictionary
        """
        # In a real implementation, this would parse the structured response
        # For now, we'll return dummy data
        return {
            "title": "Hypothesis from assumptions",
            "statement": "Statement of hypothesis from assumptions identification",
            "rationale": "Rationale for the hypothesis",
            "testability": "How this hypothesis could be tested",
            "assumptions": [
                {"statement": "Assumption 1", "sub_assumptions": ["Sub 1.1", "Sub 1.2"], "plausibility": "High"},
                {"statement": "Assumption 2", "sub_assumptions": ["Sub 2.1", "Sub 2.2"], "plausibility": "Medium"}
            ]
        }
    
    def _parse_hypothesis_from_expansion(self, response: str) -> Dict[str, Any]:
        """Parse hypothesis from research expansion response.
        
        Args:
            response: Model response string
            
        Returns:
            Structured hypothesis dictionary
        """
        # In a real implementation, this would parse the structured response
        # For now, we'll return dummy data
        return {
            "title": "Hypothesis from expansion",
            "statement": "Statement of hypothesis from research expansion",
            "rationale": "Rationale for the hypothesis",
            "testability": "How this hypothesis could be tested",
            "novel_direction": "Description of the novel research direction"
        }
    
    def _parse_multiple_hypotheses(self, response: str) -> List[Dict[str, Any]]:
        """Parse multiple hypotheses from model response.
        
        Args:
            response: Model response string
            
        Returns:
            List of structured hypothesis dictionaries
        """
        # In a real implementation, this would parse the structured response
        # For now, we'll return dummy data
        return [
            {
                "title": "Hypothesis 1",
                "statement": "Statement of hypothesis 1",
                "rationale": "Rationale for hypothesis 1",
                "testability": "How hypothesis 1 could be tested",
                "implications": "Implications of hypothesis 1"
            },
            {
                "title": "Hypothesis 2",
                "statement": "Statement of hypothesis 2",
                "rationale": "Rationale for hypothesis 2",
                "testability": "How hypothesis 2 could be tested",
                "implications": "Implications of hypothesis 2"
            }
        ]