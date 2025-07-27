import os
import numpy as np
from PIL import Image
from lark import Lark
import json
import pickle
from td.environments.tinysvgoffset_primitives import TinySVGOffset_primitives
from parser import parse_answer_expression

def create_image_from_expression(expression, env, output_path, filename):
    try:
        # Parse the expression using the grammar's built-in parse method
        parsed_tree = env.grammar.parse(expression)
        
        # Compile the parsed tree to get the image array
        image_array = env.compiler.compile(parsed_tree)
        
        # Convert from float [0,1] to uint8 [0,255]
        image_array = (image_array * 255).astype(np.uint8)
        
        # Create PIL Image
        pil_image = Image.fromarray(image_array)
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Save the image
        image_path = os.path.join(output_path, f"{filename}.png")
        pil_image.save(image_path)
        
        print(f"Successfully saved: {image_path}")
        return image_path
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None


def generate_all_images(target_dict, env, output_base_dir="generated_images"):
    
    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Track statistics
    total_expressions = 0
    successful_generations = 0
    
    # Process each graph
    for graph_name, graph_data in target_dict.items():
        print(f"\nProcessing {graph_name}...")
        
        # Create subdirectory for this graph
        graph_output_dir = os.path.join(output_base_dir, graph_name)
        
        # Process each expression in this graph
        for exp_name, exp_data in graph_data.items():
            total_expressions += 1
            target_expression = exp_data['target_expression']
            
            # Create filename
            filename = f"{graph_name}_{exp_name}"
            
            # Generate and save image
            result = create_image_from_expression(
                target_expression, 
                env, 
                graph_output_dir, 
                filename
            )
            
            if result:
                successful_generations += 1
                # Optionally update the dictionary with the image path
                exp_data['target_image_path'] = result
    
    print(f"\n=== Generation Complete ===")
    print(f"Total expressions: {total_expressions}")
    print(f"Successfully generated: {successful_generations}")
    print(f"Failed: {total_expressions - successful_generations}")
    print(f"Images saved to: {output_base_dir}")


def generate_all_images_2(target_dict, env, output_base_dir="generated_images"):
    
    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Track statistics
    total_expressions = 0
    successful_generations = 0
    
    # Process each graph
    for i in range(len(target_dict)):
        total_expressions += 1

        instance = target_dict[i]

        print(f"\nProcessing {instance['image_path']}...")
        
        # Create subdirectory for this graph
        graph_output_dir = output_base_dir
        
        vlm_expression = parse_answer_expression(instance['reconstruction'])
        
        # Create filename
        filename = f"instance_{i+1}"
        
        # Generate and save image
        result = create_image_from_expression(
            vlm_expression, 
            env, 
            graph_output_dir, 
            filename
        )
        
        if result:
            successful_generations += 1
    
    print(f"\n=== Generation Complete ===")
    print(f"Total expressions: {total_expressions}")
    print(f"Successfully generated: {successful_generations}")
    print(f"Failed: {total_expressions - successful_generations}")
    print(f"Images saved to: {output_base_dir}")


def save_updated_dictionary(target_dict, output_path="updated_target_dict.json"):
    try:
        with open(output_path, 'w') as f:
            json.dump(target_dict, f, indent=2)
        print(f"Updated dictionary saved to: {output_path}")
    except Exception as e:
        print(f"Error saving dictionary: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Initialize the environment
    env = TinySVGOffset_primitives()

    """
    filepath = "./assets/updated_graph_silhouettes.pkl"
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ Successfully loaded: {filepath}")
    except Exception as e:
        print(f"❌ Failed to load {filepath}: {e}")
    """

    with open("./vlm_results/api_results.json", 'r') as f:
        data = json.load(f)
    print(data)
    # Generate all images
    generate_all_images_2(data, env, output_base_dir="claude_4")
    
    # Optionally save the updated dictionary
    # save_updated_dictionary(data)
