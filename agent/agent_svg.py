import logging
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .memory import Memory, State
from .prompts_svg import LLM_grammar_sys, LLM_program_synthesis_prompt, LLM_CANDIDATE_GENERATION_PROMPT
from .prompts import VLM_edits_sys, VLM_edits_user_2, VLM_scene_description_prompt, VLM_edits_with_feedback_prompt
from .api_call_gpt import call_llm, call_vlm
from .parser import parse_answer, parse_answer_json, format_message
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
    
    def vlm_judge_similarity(self, target_image_path: str, current_image_path: str) -> Tuple[str, float]:
        """
        Judge the similarity of the generated image with the target image.
        
        Args:
            target_image_path: Path to the target image
            current_image_path: Path to the current image

        Returns:
            A tuple containing the similarity judgment and the IoU score
        """

        user_prompt = """
            You are an expert SVG optimization advisor. Compare the TARGET image with the CURRENT image and provide specific, actionable feedback for improving the current image to better match the target.

            Focus on providing linguistic descriptions that can guide code-level optimizations:

            VISUAL DISCREPANCIES:
            Describe what you observe that differs between the images. Be specific about:
            - Which shapes are incorrect, missing, or malformed
            - Where elements are mispositioned (e.g., "the circle is 20px too far left")
            - Color mismatches (e.g., "the rectangle should be #FF5733 instead of #FF0000")
            - Size issues (e.g., "the text is approximately 30% too small")

            GEOMETRIC ISSUES:
            - Are curves and paths following the correct trajectories?
            - Are angles and rotations accurate?
            - Do proportional relationships between elements match?

            STYLING PROBLEMS:
            - Are stroke widths, dash patterns, or line caps correct?
            - Do opacity levels and blending modes match?
            - Are fonts, text sizes, and text positioning accurate?

            LAYOUT AND COMPOSITION:
            - How do element positions compare relatively?
            - Are there alignment, spacing, or margin issues?
            - Is the overall bounding box and canvas utilization correct?

            OPTIMIZATION RECOMMENDATIONS:
            Provide specific, implementable suggestions in natural language:
            - "Move the blue rectangle 15px upward and 10px to the right"
            - "Increase the stroke width of the border from 1px to 3px"
            - "Change the circle's fill color from red to orange (#FF8C00)"
            - "Rotate the arrow element 45 degrees clockwise"
            - "Reduce the font size from 16px to 12px"

            PRIORITY FIXES:
            List the 3 most critical changes needed, in order of visual impact.

            SEMANTIC UNDERSTANDING:
            If the SVG represents something specific (icon, diagram, illustration), comment on whether the current version maintains the semantic meaning and visual intent of the target.

            Respond in clear, direct language that a developer can immediately act upon to modify SVG code or generation parameters.
            """
        # Format the message for VLM call
        messages = format_message(user_prompt=user_prompt)
        
        # Call VLM with both images (target first, then current)
        response = call_vlm(
            messages, 
            image_paths=[target_image_path, current_image_path], 
            model_name=self.model_name
        )
        
        # Log the response
        logging.info(f"VLM Judge Response: {response}")
        
        # Parse the linguistic judgment
        linguistic_judgment = parse_answer(response)
        
        # Calculate IoU score separately using traditional computer vision
        iou_score = compute_iou(current_image_path, target_image_path)
        
        return linguistic_judgment, iou_score

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