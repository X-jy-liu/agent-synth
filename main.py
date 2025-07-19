from render_engine import render_program
from tree_editor import apply_edit
# from vlm_agent import call_vlm_agent
from pathlib import Path
from PIL import ImageChops
import matplotlib.pyplot as plt
from vlm_agent import call_vlm_agent_with_reflexion  # <-- new
import json

from utils import compute_iou

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)


# === Ground Truth Program ===
gt_program = {
    "type": "Add",
    "children": [
        {"type": "Circle", "x": 30, "y": 30, "r": 10},
        {"type": "Square", "x": 60, "y": 60, "size": 15}
    ]
}

# === Initial (Noised) Program ===
current_program = {
    "type": "Add",
    "children": [
        {"type": "Circle", "x": 40, "y": 40, "r": 5}
    ]
}

# === Render GT ===
gt_img = render_program(gt_program)

# === Edit Loop ===
max_steps = 50

prev_iou = 0.0
for step in range(max_steps):
    print(f"\n--- Step {step+1} ---")
    current_img = render_program(current_program)
    iou = compute_iou(current_img, gt_img)
    print(f"IoU: {iou:.3f}")

    if iou > 0.95:
        print("✅ IoU high enough — match found!")
        break

    try:
        result = call_vlm_agent_with_reflexion(
            current_img=current_img,
            target_img=gt_img,
            program_code=current_program,
            prev_iou=prev_iou,
            current_iou=iou
        )
        thought, edit = result["thought"], result["edit"]
        print("Agent thought:", thought)
        print("Agent edit:", edit)

        current_program = apply_edit(current_program, edit)
        prev_iou = iou  # update for next round
    except Exception as e:
        print("⚠️ Agent failed:", e)
        break

    # === Save Current State ===
    # Render new image
    current_img = render_program(current_program)
    iou = compute_iou(current_img, gt_img)

    # Save everything
    step_dir = output_dir / f"step_{step}"
    step_dir.mkdir(exist_ok=True)

    # Save image
    current_img.save(step_dir / "image.png")

    # Save syntax tree
    with open(step_dir / "program.json", "w") as f:
        json.dump(current_program, f, indent=2)

    # Save agent thought and edit
    with open(step_dir / "edit.json", "w") as f:
        json.dump({
            "thought": thought,
            "edit": edit
        }, f, indent=2)

    # Save IoU score
    with open(step_dir / "iou.txt", "w") as f:
        f.write(f"{iou:.5f}\n")


# === Final Comparison ===
final_img = render_program(current_program)
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
axs[0].imshow(gt_img, cmap="gray"); axs[0].set_title("Ground Truth")
axs[1].imshow(final_img, cmap="gray"); axs[1].set_title("Final")
axs[2].imshow(ImageChops.difference(gt_img, final_img), cmap="gray"); axs[2].set_title("Diff")
for ax in axs: ax.axis("off")
plt.tight_layout()
plt.show()
