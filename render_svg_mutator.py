import json
import random
import math
import os
from typing import List, Dict, Any, Optional, Tuple
from copy import deepcopy
from render_svg import SVGAgent

class ShapeSceneGenerator:
    """Generate and mutate primitive shape scenes for VLM testing"""
    
    def __init__(self, canvas_width: int = 600, canvas_height: int = 600):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.agent = SVGAgent(canvas_width, canvas_height)
        
        # Only rectangle and ellipse shapes
        self.shape_types = ["rectangle", "ellipse"]
        
        # Define color palette
        self.colors = [
            "red", "blue", "green", "yellow", "purple", 
            "orange", "pink", "cyan", "magenta", "lime",
            "navy", "teal", "brown", "gray", "black"
        ]
    
    def create_random_shape(self, 
                          constrained: bool = False,
                          grid_snap: bool = False) -> Dict[str, Any]:
        """Create a random shape (rectangle or ellipse) with optional constraints"""
        
        shape = {
            "shape_type": random.choice(self.shape_types),
            "x": random.randint(50, self.canvas_width - 50),
            "y": random.randint(50, self.canvas_height - 50),
            "scale_x": random.randint(30, 150),
            "scale_y": random.randint(30, 150),
            "fill_color": random.choice(self.colors),
            "stroke_color": random.choice(["none"] + self.colors[:5]),
            "stroke_width": random.choice([1, 2, 3]),
            "rotation": random.randint(0, 360) if not constrained else 0,
            "opacity": random.choice([0.7, 0.8, 0.9, 1.0])
        }
        
        if constrained:
            # Make shapes more regular
            shape["scale_y"] = shape["scale_x"]  # Square/circle shapes
            shape["rotation"] = random.choice([0, 45, 90, 135, 180])
            shape["opacity"] = 1.0
        
        if grid_snap:
            # Snap to grid
            grid_size = 50
            shape["x"] = round(shape["x"] / grid_size) * grid_size
            shape["y"] = round(shape["y"] / grid_size) * grid_size
            shape["scale_x"] = round(shape["scale_x"] / 10) * 10
            shape["scale_y"] = round(shape["scale_y"] / 10) * 10
        
        return shape
    
    def create_structured_scene(self, num_shapes: int = 5) -> List[Dict[str, Any]]:
        """Create a structured scene with non-overlapping shapes"""
        shapes = []
        regions = self._divide_canvas(num_shapes)
        
        for region in regions:
            x_min, y_min, x_max, y_max = region
            shape = {
                "shape_type": random.choice(self.shape_types),
                "x": random.randint(x_min + 30, x_max - 30),
                "y": random.randint(y_min + 30, y_max - 30),
                "scale_x": random.randint(40, min(80, (x_max - x_min) // 2)),
                "scale_y": random.randint(40, min(80, (y_max - y_min) // 2)),
                "fill_color": random.choice(self.colors),
                "stroke_color": "black",
                "stroke_width": 2,
                "rotation": 0,
                "opacity": 1.0
            }
            shapes.append(shape)
        
        return shapes
    
    def _divide_canvas(self, num_regions: int) -> List[Tuple[int, int, int, int]]:
        """Divide canvas into regions for shape placement"""
        regions = []
        
        if num_regions <= 4:
            # 2x2 grid
            mid_x = self.canvas_width // 2
            mid_y = self.canvas_height // 2
            regions = [
                (0, 0, mid_x, mid_y),
                (mid_x, 0, self.canvas_width, mid_y),
                (0, mid_y, mid_x, self.canvas_height),
                (mid_x, mid_y, self.canvas_width, self.canvas_height)
            ][:num_regions]
        else:
            # 3x3 grid
            third_x = self.canvas_width // 3
            third_y = self.canvas_height // 3
            for i in range(3):
                for j in range(3):
                    if len(regions) < num_regions:
                        regions.append((
                            j * third_x, i * third_y,
                            (j + 1) * third_x, (i + 1) * third_y
                        ))
        
        return regions
    
    def create_geometric_pattern(self, pattern_type: str = "grid") -> List[Dict[str, Any]]:
        """Create a geometric pattern of shapes"""
        shapes = []
        
        if pattern_type == "grid":
            rows, cols = 4, 4
            spacing_x = self.canvas_width / (cols + 1)
            spacing_y = self.canvas_height / (rows + 1)
            
            for i in range(1, rows + 1):
                for j in range(1, cols + 1):
                    shape = {
                        "shape_type": "ellipse",  # Use ellipse for circular appearance
                        "x": j * spacing_x,
                        "y": i * spacing_y,
                        "scale_x": 30,
                        "scale_y": 30,
                        "fill_color": random.choice(["red", "blue", "green"]),
                        "stroke_color": "black",
                        "stroke_width": 1,
                        "rotation": 0,
                        "opacity": 1.0
                    }
                    shapes.append(shape)
        
        elif pattern_type == "alternating":
            rows, cols = 3, 3
            spacing_x = self.canvas_width / (cols + 1)
            spacing_y = self.canvas_height / (rows + 1)
            
            for i in range(1, rows + 1):
                for j in range(1, cols + 1):
                    # Alternate between rectangle and ellipse
                    is_rect = (i + j) % 2 == 0
                    shape = {
                        "shape_type": "rectangle" if is_rect else "ellipse",
                        "x": j * spacing_x,
                        "y": i * spacing_y,
                        "scale_x": 50,
                        "scale_y": 50 if is_rect else 40,
                        "fill_color": random.choice(self.colors),
                        "stroke_color": "black",
                        "stroke_width": 2,
                        "rotation": 0,
                        "opacity": 1.0
                    }
                    shapes.append(shape)
        
        return shapes
    
    def mutate_position(self, shape: Dict[str, Any], intensity: float = 0.1) -> Dict[str, Any]:
        """Mutate shape position with intensity 0.0 to 1.0"""
        mutated = deepcopy(shape)
        max_offset = int(min(self.canvas_width, self.canvas_height) * intensity)
        
        mutated["x"] += random.randint(-max_offset, max_offset)
        mutated["y"] += random.randint(-max_offset, max_offset)
        
        # Keep within bounds
        mutated["x"] = max(50, min(self.canvas_width - 50, mutated["x"]))
        mutated["y"] = max(50, min(self.canvas_height - 50, mutated["y"]))
        
        return mutated
    
    def mutate_scale(self, shape: Dict[str, Any], intensity: float = 0.1) -> Dict[str, Any]:
        """Mutate shape scale with intensity 0.0 to 1.0"""
        mutated = deepcopy(shape)
        scale_factor = 1 + random.uniform(-intensity, intensity)
        
        mutated["scale_x"] = int(mutated["scale_x"] * scale_factor)
        mutated["scale_y"] = int(mutated["scale_y"] * scale_factor)
        
        # Keep within reasonable bounds
        mutated["scale_x"] = max(20, min(200, mutated["scale_x"]))
        mutated["scale_y"] = max(20, min(200, mutated["scale_y"]))
        
        return mutated
    
    def mutate_color(self, shape: Dict[str, Any], intensity: float = 0.1) -> Dict[str, Any]:
        """Mutate shape colors with intensity 0.0 to 1.0"""
        mutated = deepcopy(shape)
        
        if random.random() < intensity:
            mutated["fill_color"] = random.choice(self.colors)
        
        if random.random() < intensity / 2:
            mutated["stroke_color"] = random.choice(["none"] + self.colors)
        
        return mutated
    
    def mutate_rotation(self, shape: Dict[str, Any], intensity: float = 0.1) -> Dict[str, Any]:
        """Mutate shape rotation with intensity 0.0 to 1.0"""
        mutated = deepcopy(shape)
        max_rotation = int(360 * intensity)
        mutated["rotation"] += random.randint(-max_rotation, max_rotation)
        mutated["rotation"] = mutated["rotation"] % 360
        return mutated
    
    def mutate_shape_type(self, shape: Dict[str, Any], intensity: float = 0.1) -> Dict[str, Any]:
        """Potentially change shape type between rectangle and ellipse"""
        mutated = deepcopy(shape)
        
        if random.random() < intensity:
            # Toggle between rectangle and ellipse
            mutated["shape_type"] = "ellipse" if shape["shape_type"] == "rectangle" else "rectangle"
        
        return mutated
    
    def mutate_scene(self, shapes: List[Dict[str, Any]], 
                    intensity: float = 0.1,
                    mutation_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Mutate an entire scene with numeric intensity control
        
        Args:
            shapes: Original shapes
            intensity: Float from 0.0 (no change) to 1.0 (maximum change)
            mutation_types: List of mutation types to apply, or None for auto-selection
        """
        
        # Clamp intensity between 0 and 1
        intensity = max(0.0, min(1.0, intensity))
        
        # Define available mutations
        all_mutations = {
            "position": self.mutate_position,
            "scale": self.mutate_scale,
            "color": self.mutate_color,
            "rotation": self.mutate_rotation,
            "shape_type": self.mutate_shape_type
        }
        
        # Auto-select mutations based on intensity if not specified
        if mutation_types is None:
            if intensity <= 0.2:
                mutation_types = ["position"]
            elif intensity <= 0.4:
                mutation_types = ["position", "scale"]
            elif intensity <= 0.6:
                mutation_types = ["position", "scale", "rotation"]
            elif intensity <= 0.8:
                mutation_types = ["position", "scale", "color", "rotation"]
            else:
                mutation_types = list(all_mutations.keys())
        
        # Apply mutations
        mutated_shapes = []
        for shape in shapes:
            mutated = deepcopy(shape)
            
            for mutation_type in mutation_types:
                if mutation_type in all_mutations:
                    mutated = all_mutations[mutation_type](mutated, intensity)
            
            mutated_shapes.append(mutated)
        
        # Add or remove shapes for high intensity
        if intensity > 0.8:
            # Randomly remove shapes
            if len(mutated_shapes) > 2 and random.random() < (intensity - 0.8) * 2:
                mutated_shapes.pop(random.randint(0, len(mutated_shapes) - 1))
            
            # Randomly add shapes
            if random.random() < (intensity - 0.8) * 2:
                mutated_shapes.append(self.create_random_shape())
        
        return mutated_shapes
    
    def add_noise_shapes(self, shapes: List[Dict[str, Any]], 
                        noise_intensity: float = 0.1,
                        noise_count: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Add noise shapes to simulate sketch imperfections
        
        Args:
            shapes: Original shapes
            noise_intensity: Float from 0.0 to 1.0 controlling noise opacity and size
            noise_count: Number of noise shapes, or None for auto
        """
        noisy_shapes = deepcopy(shapes)
        
        if noise_count is None:
            noise_count = int(3 + noise_intensity * 10)
        
        for _ in range(noise_count):
            size_base = 5 + int(15 * noise_intensity)
            noise_shape = {
                "shape_type": "ellipse",
                "x": random.randint(0, self.canvas_width),
                "y": random.randint(0, self.canvas_height),
                "scale_x": random.randint(5, size_base),
                "scale_y": random.randint(5, size_base),
                "fill_color": "gray",
                "stroke_color": "none",
                "stroke_width": 1,
                "rotation": random.randint(0, 360),
                "opacity": 0.1 + (noise_intensity * 0.3)
            }
            noisy_shapes.append(noise_shape)
        
        return noisy_shapes
    
    def save_scene_with_intensities(self, 
                                   base_name: str = "scene",
                                   intensities: List[float] = None):
        """
        Generate and save a scene with multiple mutation intensities
        
        Args:
            base_name: Base filename for outputs
            intensities: List of intensity values (0.0 to 1.0)
        """
        
        if intensities is None:
            intensities = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        # Create original scene
        original_shapes = self.create_structured_scene(5)
        
        # Save original
        self.agent.clear()
        self.agent.create_from_dict(original_shapes)
        self.agent.save_png(f"{base_name}_original.png")
        
        # Save original as JSON for reference
        with open(f"{base_name}_original.json", 'w') as f:
            json.dump(original_shapes, f, indent=2)
        
        # Create and save mutations at different intensities
        for intensity in intensities:
            mutated = self.mutate_scene(original_shapes, intensity)
            
            self.agent.clear()
            self.agent.create_from_dict(mutated)
            self.agent.save_png(f"{base_name}_mutated_{intensity:.1f}.png")
            
            # Save JSON
            with open(f"{base_name}_mutated_{intensity:.1f}.json", 'w') as f:
                json.dump(mutated, f, indent=2)
        
        # Create noisy version (simulating sketch)
        noisy = self.add_noise_shapes(
            self.mutate_scene(original_shapes, 0.3, ["position", "scale"]),
            noise_intensity=0.3
        )
        
        self.agent.clear()
        self.agent.create_from_dict(noisy)
        self.agent.save_png(f"{base_name}_sketch.png")
        
        print(f"Generated scene variations saved with prefix '{base_name}'")
        return original_shapes
    
    def create_controlled_mutation_test(self,
                                       base_shapes: Optional[List[Dict[str, Any]]] = None,
                                       mutation_type: str = "position",
                                       intensities: List[float] = None):
        """
        Create controlled tests for specific mutation types
        
        Args:
            base_shapes: Starting shapes or None to create new
            mutation_type: Type of mutation to test
            intensities: List of intensities to test
        """
        
        if intensities is None:
            intensities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        if base_shapes is None:
            # Create a simple test scene
            base_shapes = [
                {
                    "shape_type": "rectangle",
                    "x": 200,
                    "y": 200,
                    "scale_x": 80,
                    "scale_y": 120,
                    "fill_color": "blue",
                    "stroke_color": "black",
                    "stroke_width": 2,
                    "rotation": 0,
                    "opacity": 1.0
                },
                {
                    "shape_type": "ellipse",
                    "x": 400,
                    "y": 200,
                    "scale_x": 100,
                    "scale_y": 100,
                    "fill_color": "red",
                    "stroke_color": "black",
                    "stroke_width": 2,
                    "rotation": 0,
                    "opacity": 1.0
                },
                {
                    "shape_type": "rectangle",
                    "x": 300,
                    "y": 400,
                    "scale_x": 150,
                    "scale_y": 60,
                    "fill_color": "green",
                    "stroke_color": "black",
                    "stroke_width": 2,
                    "rotation": 45,
                    "opacity": 1.0
                }
            ]
        
        results = []
        for intensity in intensities:
            mutated = self.mutate_scene(base_shapes, intensity, [mutation_type])
            
            filename = f"test_{mutation_type}_{intensity:.1f}.png"
            self.agent.clear()
            self.agent.create_from_dict(mutated)
            self.agent.save_png(filename)
            
            results.append({
                "intensity": intensity,
                "filename": filename,
                "shapes": mutated
            })
            
            print(f"Saved {filename}")
        
        return results


def main():
    """Main demonstration of shape scene generation and mutation"""
    
    generator = ShapeSceneGenerator(600, 600)
    img_save_dir = "/home/jingyang/agent-synth/data/render_svg_mutator"
    os.makedirs(img_save_dir, exist_ok=True)

    # # Example 1: Generate scene with custom intensity values
    # print("Generating scenes with numeric intensity control...")
    # generator.save_scene_with_intensities(
    #     "test_scene_1",
    #     intensities=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    # )
    
    # # Example 2: Generate geometric patterns (rectangles and ellipses only)
    # print("\nGenerating geometric patterns...")
    
    # # Grid pattern
    # grid_shapes = generator.create_geometric_pattern("grid")
    # generator.agent.clear()
    # generator.agent.create_from_dict(grid_shapes)
    # generator.agent.save_png("pattern_grid.png")
    
    # # Alternating pattern
    # alt_shapes = generator.create_geometric_pattern("alternating")
    # generator.agent.clear()
    # generator.agent.create_from_dict(alt_shapes)
    # generator.agent.save_png("pattern_alternating.png")
    
    # # Example 3: Controlled mutation testing with specific intensities
    # print("\nGenerating controlled mutation tests...")
    
    # # Test each mutation type independently
    # mutation_tests = ["position", "scale", "rotation", "color", "shape_type"]
    
    # for mutation_type in mutation_tests:
    #     print(f"Testing {mutation_type} mutations...")
    #     generator.create_controlled_mutation_test(
    #         mutation_type=mutation_type,
    #         intensities=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    #     )
    
    # # Example 4: Fine-grained control demonstration
    # print("\nDemonstrating fine-grained intensity control...")
    
    # Create base scene
    base_scene = [
        {
            "shape_type": "ellipse",
            "x": 150,  # Top-left quadrant
            "y": 150,
            "scale_x": 80,  # Medium-sized circle
            "scale_y": 80,
            "fill_color": "purple",
            "stroke_color": "black",
            "stroke_width": 3,
            "rotation": 0,
            "opacity": 1.0
        },
        {
            "shape_type": "rectangle",
            "x": 450,  # Top-right quadrant
            "y": 150,
            "scale_x": 120,  # Wide rectangle
            "scale_y": 60,
            "fill_color": "orange",
            "stroke_color": "blue",
            "stroke_width": 2,
            "rotation": 45,  # Diagonal orientation
            "opacity": 1.0
        },
        {
            "shape_type": "ellipse",
            "x": 150,  # Bottom-left quadrant
            "y": 450,
            "scale_x": 100,  # Horizontal ellipse
            "scale_y": 60,
            "fill_color": "pink",
            "stroke_color": "red",
            "stroke_width": 4,
            "rotation": 30,
            "opacity": 1.0
        },
        {
            "shape_type": "rectangle",
            "x": 450,  # Bottom-right quadrant
            "y": 450,
            "scale_x": 90,  # Square-ish rectangle
            "scale_y": 90,
            "fill_color": "green",
            "stroke_color": "black",
            "stroke_width": 5,
            "rotation": 0,
            "opacity": 1.0
        }
    ]

    # save base scene
    generator.agent.clear()
    generator.agent.create_from_dict(base_scene)
    generator.agent.save_png(f"{img_save_dir}/base_scene.png")
    
    # Apply very specific intensity values
    for intensity in [0.02, 0.15, 0.33, 0.67, 0.85, 0.95]:
        mutated = generator.mutate_scene(base_scene, intensity)
        generator.agent.clear()
        generator.agent.create_from_dict(mutated)
        generator.agent.save_png(f"{img_save_dir}/fine_control_{intensity:.2f}.png")

    print("\nAll test scenes generated successfully!")


if __name__ == "__main__":
    main()