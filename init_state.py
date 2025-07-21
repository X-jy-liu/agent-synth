from render_engine import render_program
from pathlib import Path
import matplotlib.pyplot as plt
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

# === Render and Save the gt and initial images ===
output_dir = Path("readme_images")
output_dir.mkdir(parents=True, exist_ok=True)
gt_img = render_program(gt_program, (100, 100))
gt_img.save(output_dir / "ground_truth.png")
init_img = render_program(init_program, (100, 100))
init_img.save(output_dir / "initial_image.png")