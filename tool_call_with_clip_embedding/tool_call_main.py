import torch, clip, cv2
import numpy as np
from torchvision.transforms.functional import crop, resize
from PIL import Image, ImageEnhance
from pathlib import Path
import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from render_engine import render_program
from tool_call_vlm_agent import VLMEditAgent
import matplotlib.pyplot as plt

# === Config ===
beam_width = 15
max_steps = 20
iou_threshold = 0.95
output_dir = Path("tool_call_outputs")
output_dir.mkdir(exist_ok=True)

# === Ground Truth Program ===
gt_program = {
    "type": "Add",
    "children": [
        {"type": "Square", "x": 40, "y": 40, "s": 30},
        {"type": "Circle", "x": 90, "y": 90, "r": 20},
    ]
}

# === Initial (Noised) Program ===
# init_corruped_program = {
#     "type": "Add",
#     "children": [
#         {"type": "Square", "x": 50, "y": 50, "s": 30}
#     ]
# }

init_corruped_program = {}

gt_img = render_program(gt_program, size=(128, 128))
# init_img = render_program(init_program, size=(128, 128))
corrupted_img = render_program(init_corruped_program, size=(128, 128))

# initialize the vlm agent
agent = VLMEditAgent(
    vlm_type="gemini",
    device="cuda" if torch.cuda.is_available() else "cpu",
    clip_model_name="ViT-B/32",
    output_dir=output_dir
)

print("Plotting the ground truth and initial images...")
# display the gt img and init img as subplots
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gt_img)
plt.title("Ground Truth Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(corrupted_img)
plt.title("Corrupted Image")
plt.axis('off')
plt.show()


# test the phase 1 reflexion
phase_1_memory = {}
updated_program, current_img, memory= agent.reflexion_phase_1(
    memory=phase_1_memory,
    program=init_corruped_program,
    current_image=corrupted_img,
    gt_image=gt_img,
    max_attempts=3
)

print(f"Updated Program: {updated_program}")
print(f"Buffered Memory: {memory}")
import matplotlib.pyplot as plt
# display the cuurent img and gt img as subplots
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gt_img)
plt.title("Ground Truth Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(current_img)
plt.title("Current Image")
plt.axis('off')
plt.show()

