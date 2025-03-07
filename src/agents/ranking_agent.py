"""Ranking agent for evaluating and comparing research hypotheses."""

from typing import Any, Dict, List, Optional, Tuple

from .base_agent import BaseAgent

class RankingAgent(BaseAgent):
    """Agent for ranking and comparing research hypotheses.
    
    This agent uses an Elo-based tournament system to rank hypotheses
    based on pairwise comparisons through simulated scientific debates.
    """
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a ranking task.
        
        Args:
            task: Task parameters including task type
            
        Returns:
            Ranking results
        """
        task_type = task.get("task_type", "run_tournament_matches")
        
        if task_type == "run_tournament_matches":
            count = task.get("count", 5)
            return await self._run_tournament_matches(count)
        elif task_type == "update_rankings":
            return await self._update_rankings()
        else:
            return {"error": f"Unknown task type: {task_type}"}
    
    async def _run_tournament_matches(self, count: int) -> Dict[str, Any]:
        """Run a batch of tournament matches.
        
        Args:
            count: Number of matches to run
            
        Returns:
            Match results
        """
        # Get current tournament state
        tournament_state = self.get_from_context_memory("tournament_state", {})
        
        # Initialize tournament state if needed
        if not tournament_state:
            tournament_state = self._initialize_tournament()
        
        # Get hypotheses that have been reviewed and passed
        hypotheses = self.get_from_context_memory("hypotheses", [])
        reviewed = self.get_from_context_memory("reviewed_hypotheses", [])
        
        eligible_hypotheses = [
            h for h in hypotheses 
            if h.get("id") in reviewed
        ]
        
        if len(eligible_hypotheses) < 2:
            return {
                "error": "Not enough reviewed hypotheses for tournament",
                "tournament_state": tournament_state
            }
        
        # Run tournament matches
        matches = []
        for _ in range(count):
            # Select pair of hypotheses to compare
            pair = self._select_hypothesis_pair(eligible_hypotheses, tournament_state)
            if not pair:
                break
                
            h1, h2 = pair
            
            # Run the match
            match_result = await self._run_match(h1, h2)
            matches.append(match_result)
            
            # Update Elo ratings
            self._update_elo_ratings(h1["id"], h2["id"], match_result["winner"], tournament_state)
        
        # Update tournament progress
        total_possible_matches = len(eligible_hypotheses) * (len(eligible_hypotheses) - 1) / 2
        completed_matches = tournament_state.get("completed_matches", 0) + len(matches)
        tournament_state["completed_matches"] = completed_matches
        tournament_state["progress"] = min(1.0, completed_matches / max(1, total_possible_matches))
        
        # Update top ranked hypotheses
        tournament_state["top_ranked"] = self._get_top_ranked(tournament_state, hypotheses)
        
        # Save updated tournament state
        self.update_context_memory("tournament_state", tournament_state)
        
        return {
            "matches": matches,
            "tournament_state": tournament_state
        }
    
    async def _update_rankings(self) -> Dict[str, Any]:
        """Update hypothesis rankings based on current tournament state.
        
        Returns:
            Updated tournament state
        """
        tournament_state = self.get_from_context_memory("tournament_state", {})
        hypotheses = self.get_from_context_memory("hypotheses", [])
        
        # Update top ranked hypotheses
        tournament_state["top_ranked"] = self._get_top_ranked(tournament_state, hypotheses)
        
        # Save updated tournament state
        self.update_context_memory("tournament_state", tournament_state)
        
        return {"tournament_state": tournament_state}
    
    def _initialize_tournament(self) -> Dict[str, Any]:
        """Initialize the tournament state.
        
        Returns:
            New tournament state
        """
        return {
            "ratings": {},  # Hypothesis ID -> Elo rating
            "matches": [],  # List of completed matches
            "completed_matches": 0,
            "progress": 0.0,
            "top_ranked": []
        }
    
    def _select_hypothesis_pair(
        self, 
        hypotheses: List[Dict[str, Any]], 
        tournament_state: Dict[str, Any]
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Select a pair of hypotheses to compare.
        
        Args:
            hypotheses: List of eligible hypotheses
            tournament_state: Current tournament state
            
        Returns:
            Tuple of two hypotheses to compare, or None if no suitable pair
        """
        # This is a simplified implementation
        # A real implementation would use the proximity graph and prioritize
        # newer and top-ranking hypotheses
        
        # Initialize ratings for hypotheses that don't have one
        ratings = tournament_state.get("ratings", {})
        default_rating = 1200
        
        for h in hypotheses:
            if h["id"] not in ratings:
                ratings[h["id"]] = default_rating
        
        # Simple selection: pick the first two hypotheses
        # In a real implementation, this would be more sophisticated
        if len(hypotheses) >= 2:
            return hypotheses[0], hypotheses[1]
        
        return None
    
    async def _run_match(self, h1: Dict[str, Any], h2: Dict[str, Any]) -> Dict[str, Any]:
        """Run a tournament match between two hypotheses.
        
        Args:
            h1: First hypothesis
            h2: Second hypothesis
            
        Returns:
            Match result
        """
        # For top hypotheses, use multi-turn scientific debate
        # For lower-ranked, use single-turn comparison
        
        # Use ratings to determine if this should be a detailed debate
        tournament_state = self.get_from_context_memory("tournament_state", {})
        ratings = tournament_state.get("ratings", {})
        
        h1_rating = ratings.get(h1["id"], 1200)
        h2_rating = ratings.get(h2["id"], 1200)
        
        if h1_rating >= 1300 and h2_rating >= 1300:
            return await self._run_scientific_debate(h1, h2)
        else:
            return await self._run_simple_comparison(h1, h2)
    
    async def _run_scientific_debate(self, h1: Dict[str, Any], h2: Dict[str, Any]) -> Dict[str, Any]:
        """Run a multi-turn scientific debate between two hypotheses.
        
        Args:
            h1: First hypothesis
            h2: Second hypothesis
            
        Returns:
            Debate result
        """
        h1_statement = h1.get("statement", "")
        h2_statement = h2.get("statement", "")
        
        prompt = f"""
        Compare the following two research hypotheses through a scientific debate:
        
        HYPOTHESIS A:
        {h1_statement}
        
        HYPOTHESIS B:
        {h2_statement}
        
        DEBATE FORMAT:
        
        Round 1: Initial comparison
        - Advocate for Hypothesis A: [Present strengths of A and potential weaknesses of B]
        - Advocate for Hypothesis B: [Present strengths of B and potential weaknesses of A]
        
        Round 2: Response to critiques
        - Advocate for Hypothesis A: [Address critiques and reinforce merits]
        - Advocate for Hypothesis B: [Address critiques and reinforce merits]
        
        Round 3: Synthesis and final arguments
        - Advocate for Hypothesis A: [Final argument for why A is superior]
        - Advocate for Hypothesis B: [Final argument for why B is superior]
        
        DECISION:
        Based on the debate, which hypothesis is superior in terms of:
        1. Novelty
        2. Correctness
        3. Testability
        4. Alignment with research goal
        
        Provide a clear winner (A or B) with detailed reasoning.
        """
        
        response = await self._call_model(prompt)
        
        # In a real implementation, parse the structured debate result
        # For now, randomly select a winner
        import random
        winner = random.choice([h1["id"], h2["id"]])
        
        return {
            "hypothesis_1": h1["id"],
            "hypothesis_2": h2["id"],
            "debate_summary": f"Model response to: {prompt[:30]}...",
            "winner": winner,
            "reasoning": "Reasoning for the winner selection"
        }
    
    async def _run_simple_comparison(self, h1: Dict[str, Any], h2: Dict[str, Any]) -> Dict[str, Any]:
        """Run a simple comparison between two hypotheses.
        
        Args:
            h1: First hypothesis
            h2: Second hypothesis
            
        Returns:
            Comparison result
        """
        h1_statement = h1.get("statement", "")
        h2_statement = h2.get("statement", "")
        
        prompt = f"""
        Compare the following two research hypotheses:
        
        HYPOTHESIS A:
        {h1_statement}
        
        HYPOTHESIS B:
        {h2_statement}
        
        For each hypothesis, assess:
        1. Novelty
        2. Correctness
        3. Testability
        4. Alignment with research goal
        
        Then decide which hypothesis is superior overall. Provide a clear winner (A or B) with reasoning.
        """
        
        response = await self._call_model(prompt)
        
        # In a real implementation, parse the structured comparison result
        # For now, randomly select a winner
        import random
        winner = random.choice([h1["id"], h2["id"]])
        
        return {
            "hypothesis_1": h1["id"],
            "hypothesis_2": h2["id"],
            "comparison_summary": f"Model response to: {prompt[:30]}...",
            "winner": winner,
            "reasoning": "Reasoning for the winner selection"
        }
    
    def _update_elo_ratings(
        self, 
        h1_id: str, 
        h2_id: str, 
        winner_id: str, 
        tournament_state: Dict[str, Any]
    ) -> None:
        """Update Elo ratings based on match result.
        
        Args:
            h1_id: ID of first hypothesis
            h2_id: ID of second hypothesis
            winner_id: ID of the winning hypothesis
            tournament_state: Current tournament state
        """
        ratings = tournament_state.get("ratings", {})
        
        # Ensure both hypotheses have ratings
        if h1_id not in ratings:
            ratings[h1_id] = 1200
        if h2_id not in ratings:
            ratings[h2_id] = 1200
        
        # Calculate Elo rating updates
        k_factor = 32  # Standard K-factor
        
        # Get current ratings
        r1 = ratings[h1_id]
        r2 = ratings[h2_id]
        
        # Calculate expected scores
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
        
        # Calculate actual scores
        s1 = 1 if winner_id == h1_id else 0
        s2 = 1 if winner_id == h2_id else 0
        
        # Update ratings
        ratings[h1_id] = round(r1 + k_factor * (s1 - e1))
        ratings[h2_id] = round(r2 + k_factor * (s2 - e2))
        
        # Update tournament state
        tournament_state["ratings"] = ratings
        
        # Add match to history
        matches = tournament_state.get("matches", [])
        matches.append({
            "hypothesis_1": h1_id,
            "hypothesis_2": h2_id,
            "winner": winner_id,
            "timestamp": "2025-03-07"  # In a real implementation, use actual timestamp
        })
        tournament_state["matches"] = matches
    
    def _get_top_ranked(
        self, 
        tournament_state: Dict[str, Any],
        hypotheses: List[Dict[str, Any]]
    ) -> List[str]:
        """Get the list of top-ranked hypothesis IDs.
        
        Args:
            tournament_state: Current tournament state
            hypotheses: List of all hypotheses
            
        Returns:
            List of hypothesis IDs sorted by rating
        """
        ratings = tournament_state.get("ratings", {})
        
        # Filter to only include hypotheses that exist
        hypothesis_ids = {h["id"] for h in hypotheses}
        valid_ratings = {h_id: rating for h_id, rating in ratings.items() if h_id in hypothesis_ids}
        
        # Sort by rating (descending)
        sorted_ids = sorted(valid_ratings.keys(), key=lambda h_id: valid_ratings[h_id], reverse=True)
        
        return sorted_ids