from pathlib import Path
import json
import matplotlib.pyplot as plt
from PIL import ImageChops

from render_engine import render_program
from tree_editor import apply_edit
from vlm_agent import call_vlm_agent_with_reflexion
from utils import compute_iou

# === Config ===
beam_width = 15
max_steps = 20
iou_threshold = 0.95
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
init_program = {
    "type": "Add",
    "children": [
        {"type": "Circle", "x": 40, "y": 40, "r": 5}
    ]
}

# === Initial State ===
gt_img = render_program(gt_program)
init_img = render_program(init_program)
init_iou = compute_iou(init_img, gt_img)

beam = [{
    "program": init_program,
    "iou": init_iou,
    "history": []
}]
best_overall = beam[0]
global_best_iou = init_iou

# === Beam Search Loop ===
for step in range(max_steps):
    print(f"\n=== Step {step+1} (Beam Search) ===")
    candidates = []

    for i, state in enumerate(beam):
        program = state["program"]
        prev_iou = state["iou"]
        current_img = render_program(program)
        iou = compute_iou(current_img, gt_img)

        print(f" Beam[{i}] IoU = {iou:.3f}")

        if iou > iou_threshold:
            print("✅ Beam hit target IoU!")
            best_overall = state
            break

        try:
            result = call_vlm_agent_with_reflexion(
                current_img=current_img,
                target_img=gt_img,
                program_code=program,
                prev_iou=prev_iou,
                current_iou=iou
            )
            edit = result["edit"]
            thought = result["thought"]
            print(f" Beam[{i}] Edit: {json.dumps(edit, indent=2)}")
            print(f" Beam[{i}] Thought: {thought}")

            new_program = apply_edit(program, edit)
            new_img = render_program(new_program)
            new_iou = compute_iou(new_img, gt_img)

            candidates.append({
                "program": new_program,
                "iou": new_iou,
                "history": state["history"] + [{
                    "edit": edit,
                    "thought": thought,
                    "iou": new_iou
                }]
            })
        except Exception as e:
            print(f"⚠️ Beam[{i}] failed:", e)
            continue

    if any(c["iou"] > iou_threshold for c in candidates):
        best_overall = max(candidates, key=lambda x: x["iou"])
        print("🎯 Found match in expanded beam.")
        break

    beam = sorted(candidates, key=lambda x: -x["iou"])[:beam_width]
    if not beam:
        print("❌ All beams failed.")
        break

    # Save top candidate of the step
    if beam[0]["iou"] > global_best_iou:
        global_best_iou = beam[0]["iou"]
        print(f"🌟 New global best IoU: {global_best_iou:.3f}")
        best_step = step + 1
        best_iou = beam[0]["iou"]
        best = beam[0]
    else:
        best = beam[0]  # fallback to current top beam candidate, still a full dict

    step_dir = output_dir / f"step_{step}"
    step_dir.mkdir(exist_ok=True)
    render_program(best["program"]).save(step_dir / "image.png")
    with open(step_dir / "program.json", "w") as f:
        json.dump(best["program"], f, indent=2)
    with open(step_dir / "iou.txt", "w") as f:
        f.write(f"{best['iou']:.5f}\n")
    with open(step_dir / "edit.json", "w") as f:
        json.dump(best["history"][-1] if best["history"] else {}, f, indent=2)

    if best["iou"] > best_overall["iou"]:
        best_overall = best

# === results ===
print(f"\n=== Best Overall Result ===")
print(f"Best IoU: {best_overall['iou']:.3f} at step {best_step}")
print(f"Program: {json.dumps(best_overall['program'], indent=2)}")

# === Final Visualization ===
final_img = render_program(best_overall["program"])
diff_img = ImageChops.difference(gt_img, final_img)

fig, axs = plt.subplots(1, 3, figsize=(9, 3))
axs[0].imshow(gt_img, cmap="gray")
axs[0].set_title("Ground Truth")
axs[1].imshow(final_img, cmap="gray")
axs[1].set_title("Best Candidate")
axs[2].imshow(diff_img, cmap="gray")
axs[2].set_title("Difference")
for ax in axs:
    ax.axis("off")
plt.tight_layout()
plt.show()
