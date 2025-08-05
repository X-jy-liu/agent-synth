import torch, clip, cv2
import numpy as np
from torchvision.transforms.functional import crop, resize
from PIL import Image, ImageEnhance
from pathlib import Path
import matplotlib.patches as patches
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
output_dir = Path("images")
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
# save the images
# gt_img.save(output_dir / "gt_image.png")
# corrupted_img.save(output_dir / "corrupted_image.png")

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
plt.imshow(gt_img, cmap='gray')
plt.title("Ground Truth Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(corrupted_img, cmap='gray')
plt.title("Initial Corrupted Image")
plt.axis('off')
plt.show()

# # test phase 1: primitive shape alignment reflexion
# phase_1_memory = {}
# updated_program, current_img, memory= agent.reflexion_phase_1(
#     memory=phase_1_memory,
#     program=init_corruped_program,
#     current_image=corrupted_img,
#     gt_image=gt_img,
#     max_attempts=3
# )

# print(f"Updated Program: {updated_program}")
# print(f"Buffered Memory: {memory}")
# # display the cuurent img and gt img as subplots
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(gt_img, cmap='gray')
# plt.title("Ground Truth Image")
# plt.axis('off')
# plt.subplot(1, 2, 2)
# plt.imshow(current_img, cmap='gray')
# plt.title("Current Image")
# plt.axis('off')
# plt.show()
# save_path = output_dir / "current_image.png"
# current_img.save(save_path)
# print(f"Current image saved to {save_path}")

# test the bounding box agent
current_img = Image.open(output_dir / "current_image.png").convert("L")
print(f"image size: {current_img.size}")
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(current_img, cmap='gray')
plt.title("Current Image")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(gt_img, cmap='gray')
plt.title("Ground Truth Image")
plt.axis('off')
plt.show()
detected_bounding_boxes = agent.bounding_box_edit(
    candidate_image=current_img,
    gt_image=gt_img,
    image_size=(128, 128)
)
print(f"Detected Bounding Boxes (normalized): {detected_bounding_boxes}")

# Plot with bounding boxes
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

def draw_bboxes(ax, img, bboxes, title):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

    img_w, img_h = img.size

    for box in bboxes:
        bbox = box["bbox"]
        shape_type = box.get("type", "")

        # bbox = [x, y, width, height] in normalized coordinates
        x = bbox[0] * img_w
        y = bbox[1] * img_h
        w = bbox[2] * img_w
        h = bbox[3] * img_h

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

        # Optional: label the shape type
        ax.text(x, y , shape_type, color='lime', fontsize=6, weight='bold')

# Draw on candidate
draw_bboxes(axes[0], current_img, detected_bounding_boxes.get("candidate", []), "Candidate Image")

# Draw on ground truth
draw_bboxes(axes[1], gt_img, detected_bounding_boxes.get("ground_truth", []), "Ground Truth Image")

plt.tight_layout()
plt.show()

