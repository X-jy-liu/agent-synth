from pathlib import Path
import json
import matplotlib.pyplot as plt
from PIL import ImageChops

from render_engine import render_program
from tree_editor import apply_edit
from vlm_agent_tool_call import vlm_zoom_select, vlm_propose_edit, crop_object_region
from utils import compute_iou, crop_difference_region

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
        {"type": "Square", "x": 60, "y": 60, "size": 15}
    ]
}

# === Initial (Noised) Program ===
init_program = {
    "type": "Add",
    "children": [
        {"type": "Square", "x": 40, "y": 40, "size": 5}
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
            diff_img = crop_difference_region(current_img, gt_img)
            target_path = vlm_zoom_select(
                current_img=current_img,
                target_img=gt_img,
                diff_img=diff_img,
                program_code=program,
                prev_iou=prev_iou,
                current_iou=iou
            )

            print(f" Beam[{i}] Selected target path: {target_path}")

            current_crop, _ = crop_object_region(program, target_path, current_img)
            target_crop, _ = crop_object_region(program, target_path, gt_img)
            diff_crop, _ = crop_object_region(program, target_path, diff_img)
            # save the cropped images for debugging
            current_crop.save(output_dir / f"step_{step}" / f"current_crop_{i}.png")
            target_crop.save(output_dir / f"step_{step}" / f"target_crop_{i}.png")
            diff_crop.save(output_dir / f"step_{step}" / f"diff_crop_{i}.png")

            memory = state.get("memory", [])
            thought, edit = vlm_propose_edit(
                current_crop=current_crop,
                target_crop=target_crop,
                diff_crop=diff_crop,
                program_code=program,
                target_path=target_path,
                memory=memory
            )

            print(f" Beam[{i}] Edit: {json.dumps(edit, indent=2)}")
            print(f" Beam[{i}] Thought: {thought}")

            new_program = apply_edit(program, edit)
            new_img = render_program(new_program)
            new_iou = compute_iou(new_img, gt_img)

            if new_iou < prev_iou:
                print(f"❌ Rejecting edit: IoU dropped from {prev_iou:.3f} to {new_iou:.3f}")
                continue

            new_memory = memory[-2:] + [{
                "thought": thought,
                "edit": edit,
                "prev_iou": iou,
                "current_iou": new_iou,
                "target_path": target_path
            }]

            candidates.append({
                "program": new_program,
                "iou": new_iou,
                "history": state["history"] + [{
                    "edit": edit,
                    "thought": thought,
                    "iou": new_iou
                }],
                "memory": new_memory
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
        print("❌ All edits reduced IoU — restoring previous best.")
        beam = [best]
        continue

    if beam[0]["iou"] > global_best_iou:
        global_best_iou = beam[0]["iou"]
        print(f"🌟 New global best IoU: {global_best_iou:.3f}")
        best_step = step + 1
        best_iou = beam[0]["iou"]
        best = beam[0]
    else:
        best = beam[0]

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

# === save results ===
output_file = output_dir / "final_result.json"
with open(output_file, "w") as f:
    json.dump(
    {
        "best_iou": best_overall["iou"],
        "best_step": best_step,
        "program": best_overall["program"]
    }, f, indent=2)

print(f"\nResults saved to {output_file}")
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
final_img.save(output_dir / "final_image.png")
diff_img.save(output_dir / "final_diff.png")
plt.show()