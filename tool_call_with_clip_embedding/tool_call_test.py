import torch, clip, cv2
import numpy as np
from torchvision.transforms.functional import crop, resize
from PIL import Image, ImageEnhance
from pathlib import Path
from render_engine import render_program

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
        {"type": "Square", "x": 60, "y": 60, "size": 30}
    ]
}

# === Initial (Noised) Program ===
init_program = {
    "type": "Add",
    "children": [
        {"type": "Square", "x": 50, "y": 50, "size": 30}
    ]
}

gt_img = render_program(gt_program, size=(128, 128))
init_img = render_program(init_program, size=(128, 128))

# save the rendered images
gt_img.save("test_gt_image.png")
init_img.save("test_init_image.png")

# build the pipeline to optimize from the initial program towards the ground truth program via iteratively comparing the rendered images and edit the program

# === Step 0: detect what shapes are in the ground truth image; apply edits to make sure there are the same primitives in the programs ===

# === Step 1: Add bounding boxes to the objects in gt image and rendered image from the current program 
# which is assumed to have the primitive and the agent only needs to edit the size and position ===

# === Step 2: Move the primitives in the current program until all the primitives overlap with the ones in the ground truth program (could be done implicitly)===

# === Step 3 (CLIP-based perceptual diff) Tool Calling ====
# Replaces RGB diff-mask with a model-centred similarity map.

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

@torch.no_grad()
def clip_embed(pil_img: Image.Image) -> torch.Tensor:
    """Return L2-normalised CLIP image embedding (1×D)."""
    x = clip_preprocess(pil_img).unsqueeze(0).to(device)
    z = clip_model.encode_image(x)
    return z / z.norm(dim=-1, keepdim=True)

def compute_clip_heatmap(candidate_img: Image.Image,
                         gt_img: Image.Image,
                         gt_bbox: tuple[int, int, int],      # (x, y, size)
                         grid_hw: int = 8) -> np.ndarray:
    """
    Returns an (grid_hw × grid_hw) numpy array where each cell is
    cosine-similarity of the cell patch to the GT square patch.
    """
    x_gt, y_gt, sz = gt_bbox
    gt_patch = crop(gt_img, y_gt, x_gt, sz, sz)
    z_gt = clip_embed(gt_patch)                    # 1 × D

    stride = (candidate_img.size[0] - sz) // (grid_hw - 1)
    sims = np.zeros((grid_hw, grid_hw), dtype=np.float32)

    for gy in range(grid_hw):
        for gx in range(grid_hw):
            y0 = gy * stride
            x0 = gx * stride
            patch = crop(candidate_img, y0, x0, sz, sz)
            z = clip_embed(patch)                  # 1 × D
            sims[gy, gx] = (z @ z_gt.T).item()     # cosine similarity

    # normalise to 0-1 for visualisation
    sims -= sims.min()
    sims /= sims.max() + 1e-6
    return sims                                   # (grid_hw, grid_hw)

def overlay_heatmap(base_img: Image.Image,
                    heatmap: np.ndarray,
                    alpha: float = 0.55) -> Image.Image:
    """
    Upsamples the heatmap, converts to a colormap, and overlays on base_img.
    """
    heat = cv2.resize(heatmap, base_img.size, interpolation=cv2.INTER_NEAREST)
    heat_color = cv2.applyColorMap((255 * (1 - heat)).astype(np.uint8),
                                   cv2.COLORMAP_JET)          # cool→hot
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    heat_pil = Image.fromarray(heat_color)
    # Increase brightness so dark regions are more visible
    heat_pil = ImageEnhance.Brightness(heat_pil).enhance(1.4)

    return Image.blend(base_img.convert("RGB"), heat_pil, alpha)

# --- figure out the GT square’s bbox from the program ---------------
# (If you already store this elsewhere, just pass it in.)
x_gt, y_gt, sz_gt = gt_program["children"][0]["x"], \
                    gt_program["children"][0]["y"], \
                    gt_program["children"][0]["size"]

heat = compute_clip_heatmap(init_img, gt_img,
                            gt_bbox=(x_gt, y_gt, sz_gt),
                            grid_hw=8)

overlay_img = overlay_heatmap(init_img, heat)      # pretty picture for the VLM
overlay_img.save(output_dir / "step3_clip_overlay.png")

# === Step 4: Reflexion Module: reasoning on the color-coded diff region to propose the edit with quantitative explanation. 
# While doing the reflection, it also needs to return the quantitative feedback ===