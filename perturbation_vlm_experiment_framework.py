# VLM Corruption Detection Experiment Framework
# For use in Jupyter Notebook

import json
import random
import math
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from copy import deepcopy
import matplotlib.pyplot as plt
from pathlib import Path
from render_svg_mutator import ShapeSceneGenerator

# Assuming you have the ShapeSceneGenerator class available
# from your_module import ShapeSceneGenerator

class VLMExperimentFramework:
    """Framework for conducting VLM corruption detection experiments"""
    
    def __init__(self, output_dir: str = "vlm_experiments", canvas_size: int = 600):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.canvas_size = canvas_size
        self.generator = ShapeSceneGenerator(canvas_size, canvas_size)
        
        # Experiment tracking
        self.experiments = []
        self.current_experiment_id = 0
        
    def generate_random_base_scene(self, scene_type: str = "random") -> List[Dict[str, Any]]:
        """Generate a random base scene for experiments"""
        
        if scene_type == "structured":
            # Structured scene with 3-5 shapes
            num_shapes = random.randint(3, 5)
            return self.generator.create_structured_scene(num_shapes)
        
        elif scene_type == "pattern":
            # Geometric pattern
            pattern_type = random.choice(["grid", "alternating"])
            return self.generator.create_geometric_pattern(pattern_type)
        
        elif scene_type == "random":
            # Completely random scene
            num_shapes = random.randint(3, 6)
            shapes = []
            for _ in range(num_shapes):
                shape = self.generator.create_random_shape(
                    constrained=random.choice([True, False]),
                    grid_snap=random.choice([True, False])
                )
                shapes.append(shape)
            return shapes
        
        else:
            raise ValueError(f"Unknown scene_type: {scene_type}")
    
    def generate_random_intensities(self, num_mutations: int = 5, 
                                  intensity_range: Tuple[float, float] = (0.1, 0.9)) -> List[float]:
        """Generate random intensity values for mutations"""
        
        min_intensity, max_intensity = intensity_range
        intensities = []
        
        for _ in range(num_mutations):
            intensity = random.uniform(min_intensity, max_intensity)
            # Round to 2 decimal places for cleaner output
            intensity = round(intensity, 2)
            intensities.append(intensity)
        
        # Sort intensities for easier interpretation
        intensities.sort()
        
        return intensities
    
    def create_single_experiment(self, 
                               experiment_name: str = None,
                               scene_type: str = "random",
                               num_mutations: int = 5,
                               intensity_range: Tuple[float, float] = (0.1, 0.9),
                            #    add_noise: bool = True,
                               save_images: bool = True) -> Dict[str, Any]:
        """
        Create a single experiment with random base scene and random intensity mutations
        
        Args:
            experiment_name: Name for this experiment
            scene_type: Type of base scene ("random", "structured", "pattern")
            num_mutations: Number of mutated versions to create
            intensity_range: (min, max) range for random intensities
            save_images: Whether to save images to disk
            
        Returns:
            Dictionary containing experiment data
        """
        
        self.current_experiment_id += 1
        
        if experiment_name is None:
            experiment_name = f"experiment_{self.current_experiment_id:03d}"
        
        # Create experiment directory
        exp_dir = self.output_dir / experiment_name
        exp_dir.mkdir(exist_ok=True)
        
        # Generate random base scene
        base_scene = self.generate_random_base_scene(scene_type)
        
        # Generate random intensities
        intensities = self.generate_random_intensities(num_mutations, intensity_range)
        
        # Create mutations
        mutations = []
        image_paths = []
        
        # Save original scene
        if save_images:
            self.generator.agent.clear()
            self.generator.agent.create_from_dict(base_scene)
            original_path = exp_dir / f"{experiment_name}_original.png"
            self.generator.agent.save_png(str(original_path))
            image_paths.append(("original", 0.0, str(original_path)))
        
        # Create and save mutations
        for i, intensity in enumerate(intensities):
            # Apply mutations
            mutated_scene = self.generator.mutate_scene(base_scene, intensity)
            
            # # Optionally add noise
            # if add_noise and intensity > 0.3:
            #     noise_intensity = min(intensity * 0.5, 0.4)
            #     mutated_scene = self.generator.add_noise_shapes(
            #         mutated_scene, 
            #         noise_intensity=noise_intensity
            #     )
            
            mutations.append({
                "mutation_id": i + 1,
                "intensity": intensity,
                "scene_data": mutated_scene,
                # "has_noise": add_noise and intensity > 0.3
            })
            
            # Save image
            if save_images:
                self.generator.agent.clear()
                self.generator.agent.create_from_dict(mutated_scene)
                mutation_path = exp_dir / f"{experiment_name}_mutation_{i+1:02d}_intensity_{intensity:.2f}.png"
                self.generator.agent.save_png(str(mutation_path))
                image_paths.append((f"mutation_{i+1}", intensity, str(mutation_path)))
        
        # Create experiment record
        experiment_data = {
            "experiment_id": self.current_experiment_id,
            "experiment_name": experiment_name,
            "scene_type": scene_type,
            "base_scene": base_scene,
            "mutations": mutations,
            "intensities": intensities,
            "num_mutations": num_mutations,
            "intensity_range": intensity_range,
            # "add_noise": add_noise,
            "image_paths": image_paths,
            "experiment_dir": str(exp_dir)
        }
        
        # Save experiment metadata
        with open(exp_dir / f"{experiment_name}_metadata.json", 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        # Add to experiments list
        self.experiments.append(experiment_data)
        
        return experiment_data
    
    def create_batch_experiments(self, 
                               num_experiments: int = 10,
                               scene_type: str = "random",
                               batch_name: str = "batch",
                               fixed_ranges: bool = False) -> List[Dict[str, Any]]:
        """
        Create multiple experiments for batch testing
        
        Args:
            num_experiments: Number of experiments to create
            scene_type: a string representing the type of scene to create
            batch_name: Name prefix for the batch
            
        Returns:
            List of experiment data dictionaries
        """
        
        # if scene_types is None:
        #     scene_types = ["random", "structured", "pattern"]
        
        batch_experiments = []
        
        for i in range(num_experiments):
            # num_mutations = random.randint(4, 6)
            num_mutations = 5

            # # Vary intensity ranges for different difficulty levels
            # if i % 3 == 0:  # Easy: small differences
            #     intensity_range = (0.1, 0.4)
            # elif i % 3 == 1:  # Medium: moderate differences
            #     intensity_range = (0.2, 0.7)
            # else:  # Hard: large differences
            #     intensity_range = (0.4, 0.9)
            
            experiment_name = f"{batch_name}_{i+1:03d}"
            
            if fixed_ranges == True:
                experiment = self.create_single_experiment_five_ranges(
                    experiment_name=experiment_name,
                    scene_type=scene_type
                )
            else:
                experiment = self.create_single_experiment(
                    experiment_name=experiment_name,
                    scene_type=scene_type,
                    num_mutations=num_mutations,
                    intensity_range=(0.0,1.0),
                    # add_noise=random.choice([True, False])
                )
            
            batch_experiments.append(experiment)
            
            print(f"Created {experiment_name}: {scene_type} scene, "
                  f"intensities {experiment['intensities']}")
        
        return batch_experiments
    
    def create_single_experiment_five_ranges(self, 
                                       experiment_name: str = None,
                                       scene_type: str = "random",
                                       save_images: bool = True) -> Dict[str, Any]:
        """
        Create experiment with exactly five mutations from your specified ranges:
        0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
        """
        
        self.current_experiment_id += 1
        
        if experiment_name is None:
            experiment_name = f"experiment_{self.current_experiment_id:03d}"
        
        # Create experiment directory
        exp_dir = self.output_dir / experiment_name
        exp_dir.mkdir(exist_ok=True)
        
        # Generate random base scene
        base_scene = self.generate_random_base_scene(scene_type)
        
        # Generate intensities from your 5 ranges
        intensities = self.generate_five_range_intensities()
        
        # Create mutations
        mutations = []
        image_paths = []
        
        # Save original scene
        if save_images:
            self.generator.agent.clear()
            self.generator.agent.create_from_dict(base_scene)
            original_path = exp_dir / f"{experiment_name}_original.png"
            self.generator.agent.save_png(str(original_path))
            image_paths.append(("original", 0.0, str(original_path)))
        
        # Create and save mutations
        for i, intensity in enumerate(intensities):
            # Apply mutations
            mutated_scene = self.generator.mutate_scene(base_scene, intensity)
            
            mutations.append({
                "mutation_id": i + 1,
                "intensity": intensity,
                "scene_data": mutated_scene,
                "intensity_range_category": self._get_five_range_category(intensity)
            })
            
            # Save image
            if save_images:
                self.generator.agent.clear()
                self.generator.agent.create_from_dict(mutated_scene)
                mutation_path = exp_dir / f"{experiment_name}_mutation_{i+1:02d}_intensity_{intensity:.2f}.png"
                self.generator.agent.save_png(str(mutation_path))
                image_paths.append((f"mutation_{i+1}", intensity, str(mutation_path)))
        
        # Create experiment record
        experiment_data = {
            "experiment_id": self.current_experiment_id,
            "experiment_name": experiment_name,
            "scene_type": scene_type,
            "base_scene": base_scene,
            "mutations": mutations,
            "intensities": intensities,
            "num_mutations": 4,
            "intensity_ranges": ["0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9"],
            "intensity_categories": [self._get_five_range_category(i) for i in intensities],
            "image_paths": image_paths,
            "experiment_dir": str(exp_dir)
        }
        
        # Save experiment metadata
        with open(exp_dir / f"{experiment_name}_metadata.json", 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        # Add to experiments list
        self.experiments.append(experiment_data)
        
        return experiment_data
    
    def generate_five_range_intensities(self) -> List[float]:
        """
        Generate exactly 5 intensity values from your specified ranges:
        0.1-0.3, 0.3-0.5, 0.5-0.7, 0.7-0.9, 0.9-1.0

        Returns:
            List of 5 intensity values, one from each range
        """
        
        # Define the 5 ranges as you specified
        ranges = [
            (0.0, 0.2),   # Range 1
            (0.2, 0.4),   # Range 2
            (0.4, 0.6),   # Range 3
            (0.6, 0.8),   # Range 4
            (0.8, 1.0)    # Range 5
        ]
        
        intensities = []
        
        for min_val, max_val in ranges:
            # Generate random intensity within this range
            intensity = random.uniform(min_val, max_val)
            # Round to 2 decimal places
            intensity = round(intensity, 2)
            intensities.append(intensity)
        
        # Sort intensities for consistent ordering
        intensities.sort()
        
        return intensities

    def _get_five_range_category(self, intensity: float) -> str:
        """Get category name for your 5 ranges"""

        if 0.0 <= intensity < 0.2:
            return "range_1_low"
        elif 0.2 <= intensity < 0.4:
            return "range_2_medium_low"
        elif 0.4 <= intensity < 0.6:
            return "range_3_medium_high"
        elif 0.6 <= intensity < 0.8:
            return "range_4_high"
        elif 0.8 <= intensity < 1.0:
            return "range_5_very_high"
        else:
            return "unknown"

    def prepare_vlm_test_set(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for VLM testing from an experiment
        
        Returns test data in format suitable for VLM evaluation
        """
        
        test_data = {
            "experiment_id": experiment_data["experiment_id"],
            "experiment_name": experiment_data["experiment_name"],
            "images": [],
            "ground_truth": {
                "least_corrupted": "original",
                "corruption_order": ["original"] + [f"mutation_{i+1}" for i in range(len(experiment_data["mutations"]))],
                "intensities": [0.0] + experiment_data["intensities"]
            }
        }
        
        # Add image information
        for label, intensity, path in experiment_data["image_paths"]:
            test_data["images"].append({
                "label": label,
                "intensity": intensity,
                "path": path,
                "is_original": label == "original"
            })
        
        return test_data
    
    def create_pairwise_comparisons(self, experiment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create pairwise comparison tasks from an experiment
        
        Returns list of pairwise comparison tasks for VLM testing
        """
        
        comparisons = []
        images = experiment_data["image_paths"]
        
        # Generate all possible pairs
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                label1, intensity1, path1 = images[i]
                label2, intensity2, path2 = images[j]
                
                comparison = {
                    "comparison_id": f"{experiment_data['experiment_name']}_pair_{i}_{j}",
                    "image1": {"label": label1, "intensity": intensity1, "path": path1},
                    "image2": {"label": label2, "intensity": intensity2, "path": path2},
                    "correct_answer": "image1" if intensity1 < intensity2 else "image2",
                    "intensity_difference": abs(intensity2 - intensity1),
                    "difficulty": "easy" if abs(intensity2 - intensity1) > 0.4 else 
                                 "medium" if abs(intensity2 - intensity1) > 0.2 else "hard"
                }
                
                comparisons.append(comparison)
        
        return comparisons
    
    def create_multiple_choice_test(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create multiple choice test from an experiment
        
        Returns multiple choice test data for VLM evaluation
        """
        
        # Shuffle the images for multiple choice presentation
        images = experiment_data["image_paths"].copy()
        random.shuffle(images)
        
        # Find the correct answer (original image)
        correct_index = None
        for i, (label, intensity, path) in enumerate(images):
            if label == "original":
                correct_index = i
                break
        
        test = {
            "test_id": f"{experiment_data['experiment_name']}_mc",
            "experiment_name": experiment_data["experiment_name"],
            "instruction": "Select the image with the least corruption/distortion:",
            "options": [],
            "correct_answer": correct_index,
            "intensities": []
        }
        
        for i, (label, intensity, path) in enumerate(images):
            test["options"].append({
                "option_id": i,
                "label": label,
                "path": path
            })
            test["intensities"].append(intensity)
        
        return test
    
    def visualize_experiment(self, experiment_data: Dict[str, Any], figsize: Tuple[int, int] = (15, 10)):
        """
        Create a visualization of an experiment showing all images
        """
        
        images = experiment_data["image_paths"]
        num_images = len(images)
        
        # Calculate grid dimensions
        cols = min(3, num_images)
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        fig.suptitle(f"Experiment: {experiment_data['experiment_name']}", fontsize=16)
        
        for i, (label, intensity, path) in enumerate(images):
            if i < len(axes):
                # Load and display image
                try:
                    import matplotlib.image as mpimg
                    img = mpimg.imread(path)
                    axes[i].imshow(img)
                    axes[i].set_title(f"{label}\nIntensity: {intensity:.2f}")
                    axes[i].axis('off')
                except:
                    # If image loading fails, show placeholder
                    axes[i].text(0.5, 0.5, f"{label}\nIntensity: {intensity:.2f}", 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(images), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def export_experiments_summary(self, filename: str = "experiments_summary.csv") -> pd.DataFrame:
        """
        Export summary of all experiments to CSV
        """
        
        summary_data = []
        
        for exp in self.experiments:
            for i, mutation in enumerate(exp["mutations"]):
                summary_data.append({
                    "experiment_id": exp["experiment_id"],
                    "experiment_name": exp["experiment_name"],
                    "scene_type": exp["scene_type"],
                    "mutation_id": mutation["mutation_id"],
                    "intensity": mutation["intensity"],
                    "has_noise": mutation["has_noise"],
                    "num_shapes_original": len(exp["base_scene"]),
                    "num_shapes_mutated": len(mutation["scene_data"])
                })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(self.output_dir / filename, index=False)
        
        return df

# Example usage functions for Jupyter notebook
def run_single_experiment_demo():
    """Demo function for running a single experiment"""
    
    framework = VLMExperimentFramework()
    
    # Create a single experiment
    experiment = framework.create_single_experiment(
        experiment_name="demo_experiment",
        scene_type="structured",
        num_mutations=5,
        intensity_range=(0.2, 0.8)
    )
    
    print("Experiment created!")
    print(f"Base scene has {len(experiment['base_scene'])} shapes")
    print(f"Mutation intensities: {experiment['intensities']}")
    
    # Visualize the experiment
    fig = framework.visualize_experiment(experiment)
    plt.show()
    
    # Create test data for VLM
    vlm_test = framework.prepare_vlm_test_set(experiment)
    pairwise_comparisons = framework.create_pairwise_comparisons(experiment)
    multiple_choice = framework.create_multiple_choice_test(experiment)
    
    print(f"\nGenerated {len(pairwise_comparisons)} pairwise comparisons")
    print(f"Multiple choice test with {len(multiple_choice['options'])} options")
    
    return framework, experiment, vlm_test, pairwise_comparisons, multiple_choice

def run_batch_experiments_demo():
    """Demo function for running batch experiments"""
    
    framework = VLMExperimentFramework()
    
    # Create batch of experiments
    batch = framework.create_batch_experiments(
        num_experiments=5,
        scene_types=["random", "structured", "pattern"],
        batch_name="demo_batch"
    )
    
    print(f"Created {len(batch)} experiments")
    
    # Export summary
    summary_df = framework.export_experiments_summary("demo_batch_summary.csv")
    print(f"\nSummary exported with {len(summary_df)} mutation records")
    
    # Show summary statistics
    print("\nIntensity distribution:")
    print(summary_df['intensity'].describe())
    
    return framework, batch, summary_df

# Instructions for use in Jupyter notebook
print("""
VLM Experiment Framework - Usage Instructions
============================================

1. Run a single experiment:
   framework, experiment, vlm_test, pairwise, mc = run_single_experiment_demo()

2. Run batch experiments:
   framework, batch, summary = run_batch_experiments_demo()

3. Create custom experiment:
   framework = VLMExperimentFramework()
   exp = framework.create_single_experiment(
       experiment_name="my_experiment",
       scene_type="random",
       num_mutations=5,
       intensity_range=(0.1, 0.9)
   )

4. Visualize experiment:
   framework.visualize_experiment(exp)

5. Generate VLM test data:
   vlm_test = framework.prepare_vlm_test_set(exp)
   pairwise = framework.create_pairwise_comparisons(exp)
   mc = framework.create_multiple_choice_test(exp)
""")