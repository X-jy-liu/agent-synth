import json
import os
from PIL import Image
from tqdm import tqdm
import copy

from render_engine import render_program
from vlm_agent import call_vlm_agent_with_reflexion_tool_calling
from utils import compute_iou  # Assume you have one
from tree_editor import apply_edit     # Assume this updates a program dict

DATASET_PATH = "data/benchmark/dataset.json"
IMAGE_FOLDER = "data/benchmark/images"

def load_image(path):
    return Image.open(path).convert("L")

def test_agent_on_dataset(max_steps=20):
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    results = []
    best_program = None
    best_iou = -float("inf")
    for sample in tqdm(dataset, desc="Testing VLM agent"):
        sample_id = sample["id"]
        target_img = load_image(os.path.join(IMAGE_FOLDER, os.path.basename(sample["target_image"])))

        program = sample["initial_program"]
        original_program = copy.deepcopy(program)
        gt_program = sample.get("gt_program", None)

        history = []
        success = False

        for step in range(max_steps):
            current_img = render_program(program)
            iou = compute_iou(current_img, target_img)

            if iou >= best_iou:
                best_iou = iou
                best_program = copy.deepcopy(program)

            # Dummy previous IoU (or use history[-1] if multiple edits supported)
            prev_iou = history[-1]["iou"] if history else 0.0

            try:
                result = call_vlm_agent_with_reflexion_tool_calling(
                    current_img=current_img,
                    target_img=target_img,
                    program_code=program,
                    prev_iou=prev_iou,
                    current_iou=iou
                )
            except Exception as e:
                print(f"[{sample_id}] Step {step} failed: {e}")
                break

            # Apply the proposed edit
            try:
                program = apply_edit(program, result["edit"])
            except Exception as e:
                print(f"[{sample_id}] Failed to apply edit: {e}")
                break

            history.append({
                "program": program,
                "step": step,
                "iou": iou,
                "thought": result["thought"],
                "edit": result["edit"]
            })

        results.append({
            "id": sample_id,
            "success": success,
            "original_program": original_program,
            "gt_program": gt_program,
            "best_program": best_program,
            "final_iou": history[-1]["iou"] if history else 0.0,
            "steps": len(history),
            "history": history
        })

    return results

if __name__ == "__main__":
    result_log = test_agent_on_dataset(max_steps=20)
    with open("data/benchmark/test_results.json", "w") as f:
        json.dump(result_log, f, indent=2)
    print("✅ Test completed. Results saved to test_results.json")
