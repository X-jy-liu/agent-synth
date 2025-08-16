# Simple analysis: Which ranges does VLM choose as "least corrupted"?

import json
import numpy as np
from pathlib import Path
import re
from collections import Counter

def simple_vlm_choice_analysis(results_dir="perturbation_vlm_experiments"):
    """Simple analysis of VLM range choices"""
    
    print("="*60)
    print("SIMPLE VLM CHOICE ANALYSIS")
    print("="*60)
    print("Question: Which intensity ranges does VLM choose as 'least corrupted'?")
    print("(Correct answer is always Range 1: 0.0-0.2)")
    print()
    
    # Load results
    results_dir = Path(results_dir)
    # result_files = list(results_dir.glob("*/batch_02_flash_*_results.json"))
    result_files = [
        f for f in results_dir.glob("*/batch_02_*_results.json")
        if "flash" not in f.name
    ]
        
    choices = []
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract VLM choice
            vlm_path = data.get('vlm_chosen_image', '')
            match = re.search(r'intensity_([0-9]+\.[0-9]+)', vlm_path)
            
            if match:
                intensity = float(match.group(1))
                
                # Categorize into range
                if 0.0 <= intensity < 0.2:
                    range_chosen = "Range 1 (0.0-0.2) ✅"
                elif 0.2 <= intensity < 0.4:
                    range_chosen = "Range 2 (0.2-0.4) ❌"
                elif 0.4 <= intensity < 0.6:
                    range_chosen = "Range 3 (0.4-0.6) ❌"
                elif 0.6 <= intensity < 0.8:
                    range_chosen = "Range 4 (0.6-0.8) ❌"
                elif 0.8 <= intensity < 1.0:
                    range_chosen = "Range 5 (0.8-1.0) ❌"
                else:
                    continue
                
                choices.append({
                    'range': range_chosen,
                    'intensity': intensity,
                    'success': data.get('success', False),
                    'confidence': data.get('vlm_confidence', 0)
                })
                
        except Exception as e:
            continue
    
    print(f"Analyzed {len(choices)} VLM choices")
    print()
    
    # Count choices
    choice_counts = Counter([choice['range'] for choice in choices])
    
    print("VLM'S CHOICE FREQUENCY:")
    print("-" * 30)
    
    total_choices = len(choices)
    for range_name, count in choice_counts.most_common():
        percentage = count / total_choices * 100
        print(f"{range_name:25} | {count:3d} times ({percentage:5.1f}%)")
    
    print()
    
    # Error analysis
    errors = [choice for choice in choices if not choice['success']]
    error_ranges = Counter([choice['range'] for choice in errors])
    
    print("WHEN VLM MAKES MISTAKES, IT CHOOSES:")
    print("-" * 40)
    
    if errors:
        for range_name, count in error_ranges.most_common():
            percentage = count / len(errors) * 100
            print(f"{range_name:25} | {count:3d} times ({percentage:5.1f}% of errors)")
    
    # Key insights
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    
    correct_choices = choice_counts.get("Range 1 (0.0-0.2) ✅", 0)
    success_rate = correct_choices / total_choices
    
    print(f"✅ VLM chooses correctly (Range 1): {success_rate:.1%} of the time")
    print(f"❌ VLM makes mistakes: {1-success_rate:.1%} of the time")
    
    if errors:
        most_common_error = error_ranges.most_common(1)[0]
        print(f"🔴 Most common mistake: {most_common_error[0]} ({most_common_error[1]} times)")
        
        # Distance analysis
        range_2_errors = error_ranges.get("Range 2 (0.2-0.4) ❌", 0)
        distant_errors = sum(error_ranges.get(f"Range {i} ({0.2*(i-1):.1f}-{0.2*i:.1f}) ❌", 0) for i in [3,4,5])
        
        if range_2_errors + distant_errors > 0:
            adjacent_rate = range_2_errors / (range_2_errors + distant_errors)
            print(f"📊 Error pattern: {adjacent_rate:.1%} choose adjacent range (Range 2), {1-adjacent_rate:.1%} choose distant ranges")
    
    return choices, choice_counts, error_ranges

# Run the simple analysis
choices, choice_counts, error_ranges = simple_vlm_choice_analysis("perturbation_vlm_experiments")