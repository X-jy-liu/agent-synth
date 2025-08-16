import logging
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .memory import Memory, State
from .prompts_svg import LLM_grammar_sys, LLM_program_synthesis_prompt, LLM_CANDIDATE_GENERATION_PROMPT
from .prompts import VLM_edits_sys, VLM_edits_user_2, VLM_scene_description_prompt, VLM_edits_with_feedback_prompt
# from .api_call_gpt import call_llm, call_vlm
from .api_call_gemini import call_llm, call_vlm
from .parser import parse_answer, parse_answer_json, format_message, parse_between_tags
from .utils import compute_iou
from render_svg import SVGAgent


logging.basicConfig(
    filename='agent_svg.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


class Agent:
    """
    Agent that manages the iterative optimization process using VLM and LLM with candidate selection.
    """
    
    def __init__(self, model_name,
                 target_image_path: str, canvas_w=600, canvas_h=600):
        self.model_name = model_name
        self.target_image_path = target_image_path
        self.memory = Memory()
        self.SVGrender = SVGAgent(canvas_height=canvas_h, canvas_width=canvas_w)
        
        # Target scene description (set during initialization)
        self.target_scene_description: Optional[Dict[str, Any]] = None
        
        # Track feedback for VLM
        self.last_failed_suggestions: Optional[str] = None
        self.current_iou: float = 0.0
        
        # Track optimization history
        self.optimization_history: List[Dict[str, Any]] = []
    
    def initialize(self) -> str:
        logging.info("🎯 Step 1: Analyzing target image and generating initial program...")
        
        # Get target scene description from VLM
        self.target_scene_description = self._describe_scene_with_vlm(self.target_image_path)
        logging.info(f"📋 Target scene description: {len(self.target_scene_description.get('primitives', []))} primitives")
        
        # Generate initial program using LLM
        initial_expression = self._generate_initial_program(self.target_scene_description)
        logging.info(f"🔧 Initial expression: {initial_expression}...")

        current_state = State(
            current_expression=initial_expression,
            scene_description=None,
            primitive_actions=None
        )
        # Update memory with new state
        self.memory.add_state(current_state)
        
        return initial_expression
    
    def optimization_step(self, current_image_path: str, current_expression, output_path) -> Tuple[str, Dict[str, Any], bool]:
        """
        Intermediate step: One iteration of the optimization workflow with candidate selection.
        
        Args:
            current_image_path: Path to the current generated image
            
        Returns:
            Tuple of (new_expression, step_info, improvement_made)
        """
        logging.info("🔄 Optimization step starting...")
        
        # current_state = self.memory.get_current_state()
        # if not current_state:
        #     raise ValueError("No current state found. Call initialize() first.")
        
        # Calculate current IoU for comparison
        self.current_iou = compute_iou(current_image_path, self.target_image_path)
        logging.info(f"📊 Current IoU: {self.current_iou:.4f}")
        
        # Step 1: Generate modification actions (with feedback if available)
        logging.info("⚡ Step 1: Generating modification actions...")
        actions = self._generate_modification_actions_with_feedback(
            self.target_image_path, current_image_path
        )
        
        # Step 2: Generate 5 candidate expressions
        logging.info("🎲 Step 2: Generating 5 candidate expressions...")
        candidates = self._generate_candidate_expressions(
            current_expression,
            actions
        )
        
        # Step 3: Evaluate candidates and select best one
        logging.info("🏆 Step 3: Evaluating candidates and selecting best...")
        best_candidate, candidate_ious, improvement_made = self._select_best_candidate(
            candidates, output_path
        )
        
        # Step 4: Update state and feedback based on results
        step_info = {
            "actions": actions,
            "candidates": candidates,
            "candidate_ious": candidate_ious,
            "best_candidate": best_candidate,
            "improvement_made": improvement_made,
            "current_iou": self.current_iou,
            "best_candidate_iou": max(candidate_ious) if candidate_ious else self.current_iou
        }
        
        if improvement_made:
            logging.info(f"✅ Improvement found! IoU: {self.current_iou:.4f} → {max(candidate_ious):.4f}")
            new_expression = best_candidate
            self.last_failed_suggestions = None  # Reset feedback
        else:
            logging.info(f"❌ No improvement. Keeping current expression. Best candidate IoU: {max(candidate_ious):.4f}")
            new_expression = current_expression
            self.last_failed_suggestions = actions  # Store for feedback
        
        # Update memory
        current_state = State(
            current_expression=new_expression,
            scene_description=None,
            primitive_actions=[actions]
        )
        self.memory.add_state(current_state)
        
        # Track optimization history
        self.optimization_history.append(step_info)
        
        logging.info(f"🔄 Optimization step complete.")
        return new_expression, step_info, improvement_made
    
    def _describe_scene_with_vlm(self, image_path: str) -> Dict[str, Any]:
        messages = format_message(user_prompt=VLM_scene_description_prompt)
        response = call_vlm(messages, image_paths=[image_path], model_name=self.model_name)
        logging.info(response)
        scene_description = parse_answer_json(response)
        return scene_description
    
    def _generate_initial_program(self, scene_description: Dict[str, Any]) -> str:
        """Generate initial tinySVG program using LLM."""
        user_prompt = LLM_program_synthesis_prompt.format(vlm_description=scene_description)
        sys_prompt = LLM_grammar_sys
        messages = format_message(sys_prompt, user_prompt)
        response = call_llm(messages, model_name=self.model_name)
        logging.info(response)
        init_program = parse_answer_json(response)
        return init_program
    
    def _generate_modification_actions_with_feedback(self, target_image_path: str, current_image_path: str) -> str:
        """Generate modification actions using VLM, with feedback from previous failed attempts."""
        
        if self.last_failed_suggestions is None:
            # First time or previous suggestions worked
            user_prompt = VLM_edits_user_2
            sys_prompt = VLM_edits_sys
        else:
            # Previous suggestions didn't improve IoU, provide feedback
            user_prompt = VLM_edits_with_feedback_prompt.format(
                previous_suggestions=self.last_failed_suggestions
            )
            sys_prompt = VLM_edits_sys
            logging.info(f"🔄 Providing feedback to VLM about failed suggestions")
        
        messages = format_message(sys_prompt=sys_prompt, user_prompt=user_prompt)
        response = call_vlm(messages, image_paths=[target_image_path, current_image_path], model_name=self.model_name)
        logging.info(response)
        return response
    
    def _generate_candidate_expressions(self, current_expression: str, actions: str) -> List[str]:
        """Generate 5 candidate expressions using LLM."""
        
        user_prompt = LLM_CANDIDATE_GENERATION_PROMPT.format(
            current_expression=current_expression,
            current_actions=actions,
            num_candidates=5
        )
        
        sys_prompt = LLM_grammar_sys
        messages = format_message(sys_prompt, user_prompt)
        response = call_llm(messages, model_name=self.model_name)
        logging.info(response)
        
        # Parse response to get list of candidates
        candidates_data = parse_answer_json(response)
        if isinstance(candidates_data, dict) and 'candidates' in candidates_data:
            candidates = candidates_data['candidates']
        elif isinstance(candidates_data, list):
            candidates = candidates_data
        else:
            # Fallback: treat as single candidate
            candidates = [candidates_data]
        
        # Ensure we have exactly 5 candidates
        while len(candidates) < 5:
            candidates.append(current_expression)  # Fallback to current expression
        
        return candidates[:5]  # Take only first 5
    
    def _select_best_candidate(self, candidates: List[str], output_path) -> Tuple[str, List[float], bool]:
        """
        Evaluate candidates using IoU and select the best one.
        
        Returns:
            Tuple of (best_candidate, all_ious, improvement_made)
        """
        candidate_ious = []
        
        for i, candidate in enumerate(candidates):
            try:
                self.SVGrender.clear()
                self.SVGrender.create_from_dict(candidate)
                # logging.info(f"Candidate {i}: {type(candidate)}: {candidate}")
                candidate_image_path = self.SVGrender.save_png(os.path.join(output_path, f"candidate_{i}.png"))
                # Calculate IoU with target
                iou = compute_iou(candidate_image_path, self.target_image_path)
                candidate_ious.append(iou)
                
                logging.info(f"📊 Candidate {i+1} IoU: {iou:.4f}")
                
            except Exception as e:
                logging.error(f"❌ Error evaluating candidate {i+1}: {e}")
                candidate_ious.append(0.0)  # Assign lowest score on error
        
        # Find best candidate
        best_idx = candidate_ious.index(max(candidate_ious))
        best_candidate = candidates[best_idx]
        best_iou = candidate_ious[best_idx]
        
        # Check if there's improvement
        improvement_made = best_iou > self.current_iou
        
        logging.info(f"🏆 Best candidate: {best_idx+1} with IoU {best_iou:.4f}")
        
        return best_candidate, candidate_ious, improvement_made

    def vlm_judge_similarity(self, target_image_path: str, candidate_1_path: str, candidate_2_path: str) -> dict:
        """
        Judge the similarity of two candidate images against a target image for tournament selection.
        
        Args:
            target_image_path: Path to the ground truth target image
            candidate_1_path: Path to the first candidate image
            candidate_2_path: Path to the second candidate image

        Returns:
            dict: {
                'winner': str,  # Path to the winning candidate
                'scores': dict,  # Individual scores for both candidates
                'analysis': str,  # Detailed analysis text
                'confidence': float  # Confidence in the decision (0-1)
            }
        """

        user_prompt = """
            You are an expert visual similarity evaluator for SVG graphics conducting a tournament comparison. 
            Compare the TARGET image with TWO candidate images to determine which candidate is more similar to the target.

            ==== IMAGES ====
            - TARGET (Ground Truth): {target_image_path}
            - CANDIDATE 1: {candidate_1_path}
            - CANDIDATE 2: {candidate_2_path}

            ==== EVALUATION PROCESS ====

            First, describe each image focusing on:
            - Shapes (types, counts, proportions)
            - Colors (fills, strokes, gradients)
            - Stroke properties (width, style, caps)
            - Spatial layout (positions, alignment, spacing)
            - Scale and sizing relationships
            - Completeness (missing/extra elements)

            Then evaluate BOTH candidates against the target using these criteria (0-10 scale):

            **CANDIDATE 1 EVALUATION:**
            1. Shape Accuracy: [score]/10 - [observations]
            2. Color Fidelity: [score]/10 - [observations]
            3. Stroke Properties: [score]/10 - [observations]
            4. Spatial Layout: [score]/10 - [observations]
            5. Size and Scale: [score]/10 - [observations]
            6. Completeness: [score]/10 - [observations]
            
            CANDIDATE 1 OVERALL: [average]/10

            **CANDIDATE 2 EVALUATION:**
            1. Shape Accuracy: [score]/10 - [observations]
            2. Color Fidelity: [score]/10 - [observations]
            3. Stroke Properties: [score]/10 - [observations]
            4. Spatial Layout: [score]/10 - [observations]
            5. Size and Scale: [score]/10 - [observations]
            6. Completeness: [score]/10 - [observations]
            
            CANDIDATE 2 OVERALL: [average]/10

            **TOURNAMENT DECISION:**
            - Winner: CANDIDATE [1/2]
            - Score Difference: [winner_score - loser_score]
            - Confidence Level: [HIGH/MEDIUM/LOW] based on score difference
            * HIGH: Difference ≥ 2.0 points
            * MEDIUM: Difference 0.5-1.9 points  
            * LOW: Difference < 0.5 points

            **KEY DIFFERENTIATORS:**
            - [List 2-3 main reasons why the winner is better]
            - [Mention any close aspects where candidates were similar]

            **WINNER CLASSIFICATION:**
            - EXCELLENT: Overall score ≥ 9.0
            - GOOD: Overall score 7.0-8.9
            - ACCEPTABLE: Overall score 5.0-6.9
            - POOR: Overall score < 5.0

            RESPONSE FORMAT:
            <answer>CANDIDATE_{{1 or 2}}</answer>
            <candidate1_score>[numerical score]</candidate1_score>
            <candidate2_score>[numerical score]</candidate2_score>
            <confidence>[HIGH/MEDIUM/LOW]</confidence>
            <explanation>[Complete analysis as structured above]</explanation>
        """

        # Format the message for VLM call
        messages = format_message(user_prompt=user_prompt.format(
            target_image_path=target_image_path,
            candidate_1_path=candidate_1_path,
            candidate_2_path=candidate_2_path
        ))
        
        # Call VLM with all three images (target first, then candidates)
        response = call_vlm(
            messages, 
            image_paths=[target_image_path, candidate_1_path, candidate_2_path], 
            model_name=self.model_name
        )
        
        # Log the response
        logging.info(f"VLM Tournament Judge Response: {response}")
        
        try:
            # Parse the structured response
            winner_candidate = parse_answer(response)  # Should return "CANDIDATE_1" or "CANDIDATE_2"
            candidate1_score = float(parse_between_tags(response, "candidate1_score"))
            candidate2_score = float(parse_between_tags(response, "candidate2_score"))
            confidence_level = parse_between_tags(response, "confidence")
            analysis = parse_between_tags(response, "explanation")
            
            # Determine winner path and confidence score
            if winner_candidate == "CANDIDATE_1":
                winner_path = candidate_1_path
            elif winner_candidate == "CANDIDATE_2":
                winner_path = candidate_2_path
            else:
                # Fallback to score comparison
                winner_path = candidate_1_path if candidate1_score >= candidate2_score else candidate_2_path
            
            # Convert confidence level to numerical score
            confidence_map = {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.5}
            confidence_score = confidence_map.get(confidence_level, 0.5)
            
            return {
                'winner': winner_path,
                'scores': {
                    'candidate_1': candidate1_score,
                    'candidate_2': candidate2_score
                },
                'analysis': analysis,
                'confidence': confidence_score,
                'score_difference': abs(candidate1_score - candidate2_score)
            }
            
        except Exception as e:
            logging.error(f"Error parsing VLM response: {e}")
            # Fallback: return basic linguistic judgment
            linguistic_judgment = parse_answer(response)
            return {
                'winner': candidate_1_path if "1" in linguistic_judgment else candidate_2_path,
                'scores': {'candidate_1': 0.0, 'candidate_2': 0.0},
                'analysis': response,
                'confidence': 0.5,
                'score_difference': 0.0
            }


    def run_tournament(self, target_image_path: str, candidate_paths: list, log_file_path: str = None) -> dict:
        """
        Run a tournament-style comparison to find the best candidate image.
        
        Args:
            target_image_path: Path to the ground truth target image
            candidate_paths: List of paths to candidate images
            log_file_path: Optional path to save detailed tournament log (default: auto-generated)
        
        Returns:
            dict: Tournament results with winner, all scores, and tournament bracket
        """
        if len(candidate_paths) < 2:
            raise ValueError("Tournament requires at least 2 candidates")
        
        import math
        import os
        from datetime import datetime
        
        # Generate log file path if not provided
        if log_file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = f"tournament_log_{timestamp}.txt"
        
        # Initialize detailed logging
        detailed_log = []
        tournament_log = []
        current_round = candidate_paths.copy()
        round_number = 1
        
        # Start tournament log
        log_header = f"""
    {'='*80}
    VLM TOURNAMENT LOG
    {'='*80}
    Target Image: {target_image_path}
    Total Candidates: {len(candidate_paths)}
    Tournament Start Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Model: {self.model_name}
    {'='*80}

    CANDIDATE LIST:
    """
        for i, path in enumerate(candidate_paths, 1):
            log_header += f"{i:2d}. {path}\n"
        
        log_header += f"\n{'='*80}\n"
        detailed_log.append(log_header)
        
        while len(current_round) > 1:
            next_round = []
            round_log = []
            
            round_header = f"\nROUND {round_number} - {len(current_round)} candidates\n{'-'*50}\n"
            detailed_log.append(round_header)
            logging.info(f"Tournament Round {round_number}: {len(current_round)} candidates")
            
            match_number = 1
            
            # Pair up candidates for this round
            for i in range(0, len(current_round), 2):
                if i + 1 < len(current_round):
                    # Normal pairing
                    candidate_1 = current_round[i]
                    candidate_2 = current_round[i + 1]
                    
                    match_header = f"\nMATCH {match_number} (Round {round_number}):\n"
                    match_header += f"Candidate 1: {os.path.basename(candidate_1)}\n"
                    match_header += f"Candidate 2: {os.path.basename(candidate_2)}\n"
                    match_header += f"Target: {os.path.basename(target_image_path)}\n"
                    match_header += "-" * 40 + "\n"
                    detailed_log.append(match_header)
                    
                    # Judge the match
                    result = self.vlm_judge_similarity(target_image_path, candidate_1, candidate_2)
                    
                    winner = result['winner']
                    next_round.append(winner)
                    
                    # Log match results
                    match_result = f"WINNER: {os.path.basename(winner)}\n"
                    match_result += f"Candidate 1 Score: {result['scores']['candidate_1']:.2f}\n"
                    match_result += f"Candidate 2 Score: {result['scores']['candidate_2']:.2f}\n"
                    match_result += f"Score Difference: {result['score_difference']:.2f}\n"
                    match_result += f"Confidence: {result['confidence']:.2f}\n\n"
                    match_result += "VLM DETAILED ANALYSIS:\n"
                    match_result += result['analysis'] + "\n"
                    match_result += "=" * 60 + "\n"
                    detailed_log.append(match_result)
                    
                    match_info = {
                        'match_number': match_number,
                        'candidate_1': candidate_1,
                        'candidate_2': candidate_2,
                        'winner': winner,
                        'scores': result['scores'],
                        'confidence': result['confidence'],
                        'score_difference': result['score_difference'],
                        'vlm_analysis': result['analysis']
                    }
                    round_log.append(match_info)
                    
                    logging.info(f"Match {match_number}: {os.path.basename(candidate_1)} vs {os.path.basename(candidate_2)} -> Winner: {os.path.basename(winner)}")
                    match_number += 1
                    
                else:
                    # Odd number of candidates, this one advances automatically
                    bye_info = f"\nBYE: {os.path.basename(current_round[i])} advances automatically\n" + "=" * 40 + "\n"
                    detailed_log.append(bye_info)
                    next_round.append(current_round[i])
                    logging.info(f"Bye: {current_round[i]} advances automatically")
            
            tournament_log.append({
                'round': round_number,
                'matches': round_log
            })
            
            current_round = next_round
            round_number += 1
        
        # Get final evaluation of the winner
        final_winner = current_round[0]
        
        # Log final results
        final_header = f"\n{'='*80}\nFINAL RESULTS\n{'='*80}\n"
        final_header += f"TOURNAMENT CHAMPION: {os.path.basename(final_winner)}\n"
        final_header += f"Total Rounds: {round_number - 1}\n"
        final_header += f"Tournament End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        final_header += "=" * 80 + "\n"
        detailed_log.append(final_header)
        
        # Perform final evaluation against target
        if len(candidate_paths) > 1:  # Only do final evaluation if there was actual competition
            final_eval_header = "\nFINAL CHAMPION EVALUATION vs TARGET:\n" + "-" * 50 + "\n"
            detailed_log.append(final_eval_header)
            
            # Create a dummy comparison (champion vs itself) to get detailed analysis
            final_evaluation = self.vlm_judge_similarity(target_image_path, final_winner, final_winner)
            
            final_eval_result = f"Champion: {os.path.basename(final_winner)}\n"
            final_eval_result += f"Final Score: {final_evaluation['scores']['candidate_1']:.2f}/10\n\n"
            final_eval_result += "FINAL DETAILED ANALYSIS:\n"
            final_eval_result += final_evaluation['analysis'] + "\n"
            detailed_log.append(final_eval_result)
        else:
            final_evaluation = {'scores': {'candidate_1': 0.0}, 'analysis': 'Single candidate tournament'}
        
        # Write complete log to file
        try:
            with open(log_file_path, 'w', encoding='utf-8') as f:
                f.write(''.join(detailed_log))
            logging.info(f"Tournament log saved to: {log_file_path}")
        except Exception as e:
            logging.error(f"Failed to save tournament log: {e}")
        
        return {
            'champion': final_winner,
            'tournament_log': tournament_log,
            'total_rounds': round_number - 1,
            'total_candidates': len(candidate_paths),
            'final_evaluation': final_evaluation,
            'log_file_path': log_file_path,
            'detailed_log': ''.join(detailed_log)  # Include in return for immediate access
        }
    
    # define a selection method to select the most similar candidate image to the target.
    def vlm_judge_best_candidate(self, target_image_path: str, candidate_paths: List[str]) -> dict:
        """
        Judge 5 candidate images against a target image and select the best one based on content similarity.
        
        Args:
            target_image_path: Path to the ground truth target image
            candidate_paths: List of paths to candidate images (expects 5 candidates)

        Returns:
            dict: {
                'best_candidate': str,  # Path to the best candidate
                'all_scores': List[float],  # Scores for all candidates in order
                'rankings': List[int],  # Ranking positions (1=best, 5=worst)
                'analysis': str,  # Detailed analysis text
                'confidence': float,  # Confidence in the decision (0-1)
                'score_difference': float  # Difference between best and second-best
            }
        """
        
        if len(candidate_paths) != 5:
            raise ValueError(f"Expected exactly 5 candidates, got {len(candidate_paths)}")

        user_prompt = """
            You are an expert visual similarity evaluator for SVG graphics. Your task is to compare a TARGET image 
            with FIVE candidate images and determine which candidate is most similar to the target.

            ==== IMAGES ====
            - TARGET (Ground Truth): Image 1
            - CANDIDATE A: Image 2  
            - CANDIDATE B: Image 3
            - CANDIDATE C: Image 4
            - CANDIDATE D: Image 5
            - CANDIDATE E: Image 6

            ==== EVALUATION PROCESS ====

            First, describe the TARGET image in detail, focusing on:
            - Shapes (types, counts, proportions, complexity)
            - Colors (fills, strokes, gradients, color palette)
            - Stroke properties (width, style, caps, joins)
            - Spatial layout (positions, alignment, spacing, composition)
            - Scale and sizing relationships
            - Overall visual style and completeness

            Then evaluate EACH candidate against the target using these criteria (0-10 scale):

            **CANDIDATE A EVALUATION:**
            1. Shape Accuracy: [score]/10 - [detailed observations]
            2. Color Fidelity: [score]/10 - [detailed observations]  
            3. Stroke Properties: [score]/10 - [detailed observations]
            4. Spatial Layout: [score]/10 - [detailed observations]
            5. Size and Scale: [score]/10 - [detailed observations]
            6. Completeness: [score]/10 - [detailed observations]
            
            CANDIDATE A OVERALL: [average]/10

            **CANDIDATE B EVALUATION:**
            1. Shape Accuracy: [score]/10 - [detailed observations]
            2. Color Fidelity: [score]/10 - [detailed observations]
            3. Stroke Properties: [score]/10 - [detailed observations]
            4. Spatial Layout: [score]/10 - [detailed observations]
            5. Size and Scale: [score]/10 - [detailed observations]
            6. Completeness: [score]/10 - [detailed observations]
            
            CANDIDATE B OVERALL: [average]/10

            **CANDIDATE C EVALUATION:**
            1. Shape Accuracy: [score]/10 - [detailed observations]
            2. Color Fidelity: [score]/10 - [detailed observations]
            3. Stroke Properties: [score]/10 - [detailed observations]
            4. Spatial Layout: [score]/10 - [detailed observations]
            5. Size and Scale: [score]/10 - [detailed observations]
            6. Completeness: [score]/10 - [detailed observations]
            
            CANDIDATE C OVERALL: [average]/10

            **CANDIDATE D EVALUATION:**
            1. Shape Accuracy: [score]/10 - [detailed observations]
            2. Color Fidelity: [score]/10 - [detailed observations]
            3. Stroke Properties: [score]/10 - [detailed observations]
            4. Spatial Layout: [score]/10 - [detailed observations]
            5. Size and Scale: [score]/10 - [detailed observations]
            6. Completeness: [score]/10 - [detailed observations]
            
            CANDIDATE D OVERALL: [average]/10

            **CANDIDATE E EVALUATION:**
            1. Shape Accuracy: [score]/10 - [detailed observations]
            2. Color Fidelity: [score]/10 - [detailed observations]
            3. Stroke Properties: [score]/10 - [detailed observations]
            4. Spatial Layout: [score]/10 - [detailed observations]
            5. Size and Scale: [score]/10 - [detailed observations]
            6. Completeness: [score]/10 - [detailed observations]
            
            CANDIDATE E OVERALL: [average]/10

            **FINAL RANKING AND SELECTION:**
            
            Rank all candidates from best to worst:
            1. CANDIDATE [A/B/C/D/E] - [score]/10
            2. CANDIDATE [A/B/C/D/E] - [score]/10  
            3. CANDIDATE [A/B/C/D/E] - [score]/10
            4. CANDIDATE [A/B/C/D/E] - [score]/10
            5. CANDIDATE [A/B/C/D/E] - [score]/10

            **BEST CANDIDATE SELECTION:**
            - Winner: CANDIDATE [A/B/C/D/E]
            - Best Score: [score]/10
            - Score Difference from Second Place: [difference]
            - Confidence Level: [HIGH/MEDIUM/LOW] based on score difference and absolute score
            * HIGH: Best score ≥ 8.0 AND difference ≥ 1.5 points
            * MEDIUM: Best score ≥ 6.0 OR difference 0.8-1.4 points  
            * LOW: Best score < 6.0 AND difference < 0.8 points

            **KEY DIFFERENTIATORS:**
            - [List 3-4 main reasons why the winner excels over others]
            - [Mention any close competitors and what they lacked]
            - [Note any major flaws in lower-ranked candidates]

            **QUALITY ASSESSMENT:**
            - EXCELLENT: Overall score ≥ 9.0 (near-perfect match)
            - GOOD: Overall score 7.0-8.9 (strong similarity with minor differences)
            - ACCEPTABLE: Overall score 5.0-6.9 (recognizable similarity with notable differences)
            - POOR: Overall score 3.0-4.9 (some similarity but major differences)
            - VERY_POOR: Overall score < 3.0 (little to no similarity)

            RESPONSE FORMAT:
            <answer>CANDIDATE_{{A/B/C/D/E}}</answer>
            <score_a>[numerical score for candidate A]</score_a>
            <score_b>[numerical score for candidate B]</score_b>
            <score_c>[numerical score for candidate C]</score_c>
            <score_d>[numerical score for candidate D]</score_d>
            <score_e>[numerical score for candidate E]</score_e>
            <confidence>[HIGH/MEDIUM/LOW]</confidence>
            <explanation>[Complete analysis as structured above]</explanation>
        """

        # Format the message for VLM call  
        messages = format_message(user_prompt=user_prompt)
        
        # Call VLM with target image first, then all 5 candidates
        all_image_paths = [target_image_path] + candidate_paths
        response = call_vlm(
            messages, 
            image_paths=all_image_paths, 
            model_name=self.model_name
        )
        
        # Log the response
        logging.info(f"VLM Multi-Candidate Judge Response: {response}")
        
        try:
            # Parse the structured response
            winner_candidate = parse_answer(response)  # Should return "CANDIDATE_A", "CANDIDATE_B", etc.
            
            # Extract individual scores
            scores = []
            for letter in ['a', 'b', 'c', 'd', 'e']:
                try:
                    score = float(parse_between_tags(response, f"score_{letter}"))
                    scores.append(score)
                except (ValueError, TypeError):
                    logging.warning(f"Could not parse score for candidate {letter.upper()}, defaulting to 0.0")
                    scores.append(0.0)
            
            confidence_level = parse_between_tags(response, "confidence")
            analysis = parse_between_tags(response, "explanation")
            
            # Determine best candidate index and path
            candidate_mapping = {
                "CANDIDATE_A": 0, "CANDIDATE_B": 1, "CANDIDATE_C": 2, 
                "CANDIDATE_D": 3, "CANDIDATE_E": 4
            }
            
            if winner_candidate in candidate_mapping:
                best_idx = candidate_mapping[winner_candidate]
            else:
                # Fallback to highest score
                best_idx = scores.index(max(scores))
                logging.warning(f"Could not parse winner candidate, using highest score: index {best_idx}")
            
            best_candidate_path = candidate_paths[best_idx]
            
            # Calculate rankings (1=best, 5=worst)
            sorted_indices = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
            rankings = [0] * len(scores)
            for rank, idx in enumerate(sorted_indices):
                rankings[idx] = rank + 1
            
            # Calculate score difference between best and second best
            sorted_scores = sorted(scores, reverse=True)
            score_difference = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else 0.0
            
            # Convert confidence level to numerical score
            confidence_map = {"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.5}
            confidence_score = confidence_map.get(confidence_level, 0.5)
            
            logging.info(f"Best candidate: {best_candidate_path} (index {best_idx}) with score {scores[best_idx]:.2f}")
            
            return {
                'best_candidate': best_candidate_path,
                'best_candidate_index': best_idx,
                'all_scores': scores,
                'rankings': rankings,
                'analysis': analysis,
                'confidence': confidence_score,
                'confidence_level': confidence_level,
                'score_difference': score_difference,
                'winner_score': scores[best_idx],
                'candidate_mapping': {
                    f'candidate_{i}': {'path': path, 'score': score, 'rank': rank} 
                    for i, (path, score, rank) in enumerate(zip(candidate_paths, scores, rankings))
                }
            }
            
        except Exception as e:
            logging.error(f"Error parsing VLM multi-candidate response: {e}")
            # Fallback: return candidate with highest score or first one
            try:
                # Try to extract any scores we can find
                fallback_scores = []
                for letter in ['a', 'b', 'c', 'd', 'e']:
                    try:
                        score = float(parse_between_tags(response, f"score_{letter}"))
                        fallback_scores.append(score)
                    except:
                        fallback_scores.append(0.0)
                
                best_idx = fallback_scores.index(max(fallback_scores))
                
                return {
                    'best_candidate': candidate_paths[best_idx],
                    'best_candidate_index': best_idx,
                    'all_scores': fallback_scores,
                    'rankings': list(range(1, 6)),  # Default rankings
                    'analysis': response,  # Raw response as fallback
                    'confidence': 0.5,
                    'confidence_level': 'LOW',
                    'score_difference': 0.0,
                    'winner_score': fallback_scores[best_idx],
                    'candidate_mapping': {
                        f'candidate_{i}': {'path': path, 'score': score, 'rank': i+1} 
                        for i, (path, score) in enumerate(zip(candidate_paths, fallback_scores))
                    }
                }
            except:
                # Ultimate fallback: return first candidate
                return {
                    'best_candidate': candidate_paths[0],
                    'best_candidate_index': 0,
                    'all_scores': [0.0] * 5,
                    'rankings': list(range(1, 6)),
                    'analysis': response,
                    'confidence': 0.3,
                    'confidence_level': 'LOW',
                    'score_difference': 0.0,
                    'winner_score': 0.0,
                    'candidate_mapping': {
                        f'candidate_{i}': {'path': path, 'score': 0.0, 'rank': i+1} 
                        for i, path in enumerate(candidate_paths)
                    }
                }

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of the current memory state."""
        return {
            "total_states": self.memory.size(),
            "current_expression": self.memory.get_current_state().current_expression if self.memory.size() > 0 else None,
            "target_primitives": len(self.target_scene_description.get('primitives', [])) if self.target_scene_description else 0,
            "current_iou": self.current_iou,
            "optimization_steps": len(self.optimization_history),
            "improvements_made": sum(1 for step in self.optimization_history if step.get("improvement_made", False)),
            "has_failed_suggestions": self.last_failed_suggestions is not None
        }
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get the full optimization history."""
        return self.optimization_history
    
    def reset_feedback(self):
        """Reset the VLM feedback state (useful for testing)."""
        self.last_failed_suggestions = None