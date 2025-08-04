import logging
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .memory import Memory, State
from .prompts import LLM_grammar_prompt, LLM_program_synthesis_prompt, LLM_EXPRESSION_MODIFIER_PROMPT
from .prompts import VLM_edits_sys, VLM_edits_user, VLM_scene_description_prompt
from .api_call_gpt import call_llm, call_vlm
from .parser import parse_answer, parse_answer_json, format_message


logging.basicConfig(
    filename='agent.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


class Agent:
    """
    Agent that manages the iterative optimization process using VLM and LLM.
    """
    
    def __init__(self, model_name,
                 target_image_path: str):
        self.model_name = model_name
        self.target_image_path = target_image_path
        self.memory = Memory()
        
        # Target scene description (set during initialization)
        self.target_scene_description: Optional[Dict[str, Any]] = None
    
    def initialize(self) -> str:
        """
        Step 1: Generate target scene description and initial program.
        
        Returns:
            Initial tinySVG expression
        """
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
    
    def optimization_step(self, current_image_path: str) -> Tuple[str, Dict[str, List[str]]]:
        """
        Intermediate step: One iteration of the optimization workflow.
        
        Args:
            current_image_path: Path to the current generated image
            
        Returns:
            Tuple of (new_expression, collected_actions)
        """
        logging.info("🔄 Optimization step starting...")
        
        current_state = self.memory.get_current_state()
        if not current_state:
            raise ValueError("No current state found. Call initialize() first.")
        
        
        # Step 3: For each pair, VLM generates modifying actions
        logging.info("⚡ Step 1: Generating modification actions for each primitive pair...")
            
        actions = self._generate_modification_actions(
            self.target_image_path, current_image_path
        )
        
        # Step 4: LLM gives modified expression
        logging.info("🛠️ Step 2: Generating modified expression...")
        new_expression = self._generate_modified_expression(
            current_state.current_expression,
            actions
        )
        
        current_state = State(
            current_expression=new_expression,
            scene_description=None,
            primitive_actions=[actions]
        )
        # Update memory with new state
        self.memory.add_state(current_state)
        
        logging.info(f"✅ Optimization step complete. New expression: {new_expression}...")
        return new_expression, actions
    
    def _describe_scene_with_vlm(self, image_path: str) -> Dict[str, Any]:
        
        messages = format_message(user_prompt=VLM_scene_description_prompt)
        response = call_vlm(messages, image_paths=[image_path], model_name=self.model_name)
        logging.info(response)
        scene_description = parse_answer_json(response)
        return scene_description
    
    def _generate_initial_program(self, scene_description: Dict[str, Any]) -> str:
        """Generate initial tinySVG program using LLM."""
        
        user_prompt = LLM_program_synthesis_prompt.format(vlm_description=scene_description)
        sys_prompt = LLM_grammar_prompt
        messages = format_message(sys_prompt, user_prompt)
        response = call_llm(messages, model_name=self.model_name)
        logging.info(response)
        init_program = parse_answer(response)
        return init_program
    
    def _generate_modification_actions(self, target_image_path: str, current_image_path: str) -> List[str]:
        """Generate modification actions using VLM."""
        
        user_prompt = VLM_edits_user
        
        messages = format_message(sys_prompt=VLM_edits_sys, user_prompt=user_prompt)
        response = call_vlm(messages, image_paths=[target_image_path, current_image_path], model_name=self.model_name)
        logging.info(response)
        actions = parse_answer_json(response)
        return actions
    
    def _generate_modified_expression(self, 
                                    current_expression: str,
                                    collected_actions: Dict[str, List[str]]) -> str:
        
        user_prompt = LLM_EXPRESSION_MODIFIER_PROMPT.format(
            current_expression=current_expression,
            current_actions=collected_actions
        )
        
        sys_prompt = LLM_grammar_prompt
        messages = format_message(sys_prompt, user_prompt)
        response = call_llm(messages, model_name=self.model_name)
        logging.info(response)
        new_program = parse_answer(response)
        return new_program
    
    def _find_primitive_by_id(self, scene_description: Dict[str, Any], primitive_id: str) -> Optional[Dict[str, Any]]:
        """Find a primitive by its ID in a scene description."""
        for primitive in scene_description.get('primitives', []):
            if primitive.get('id') == primitive_id:
                return primitive
        return None
    
    def _format_primitive_description(self, primitive: Dict[str, Any]) -> str:
        """Format a primitive for use in prompts."""
        features = primitive.get('features', {})
        return f"{features.get('fill_color', 'unknown')} {features.get('shape', 'shape')} in {features.get('pos_bin', 'unknown')}, {primitive.get('relative_position', 'no position info')}"

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of the current memory state."""
        return {
            "total_states": self.memory.size(),
            "current_expression": self.memory.get_current_state().current_expression if self.memory.size() > 0 else None,
            "target_primitives": len(self.target_scene_description.get('primitives', [])) if self.target_scene_description else 0
        }
