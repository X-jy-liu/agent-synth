from pathlib import Path
import clip
import torch
from PIL import Image
import json
import re
from vlm_loader import ask_claude_multi, ask_gemini_multi, ask_openai_multi
import numpy as np
from torchvision.transforms.functional import crop
from PIL import ImageEnhance
import cv2
from tree_editor import apply_edit
from render_engine import render_program
import ast
import os

class VLMEditAgent:
    def __init__(self,
                vlm_type: str,
                device: str = None,
                clip_model_name: str = "ViT-B/32",
                output_dir: str = "outputs"):
        
        self.vlm_type = vlm_type.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Store keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        # self.anthropic_api_key = anthropic_api_key
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY")

        # CLIP
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
        self.clip_model.eval()

        print(f"[INFO] Initialized with {self.vlm_type.upper()} on {self.device}")
        # Add more internal buffers if needed later
        self.memory = {}

        print(f"[INFO] Agent initialized with model: {self.vlm_type.upper()} on {self.device}")

    def vlm_ask_multi(self, images: list[Image.Image], question: str) -> str:
        if self.vlm_type == "gpt4v":
            return ask_openai_multi(images, question, self.openai_api_key)
        elif self.vlm_type == "claude3":
            return ask_claude_multi(images, question, self.anthropic_api_key)
        elif self.vlm_type == "gemini":
            return ask_gemini_multi(images, question, self.gemini_api_key)
        else:
            raise NotImplementedError(f"No VLM backend for {self.vlm_type}")

    def primitive_edit(self, candidate_image, ground_truth_image) -> list:
        """
        One-shot comparison: check primitive types and propose edits if needed.

        Returns:
            A list of proposed edits, or an empty list if no edits are needed.
        """
        question = (
            "You are a vision-based program repair agent.\n"
            "Task:\n"
            "1. You will be shown two images side-by-side:\n"
            "   - Image 1: The **candidate image** (possibly corrupted or missing shapes)\n"
            "   - Image 2: The **Ground truth image** (correct version)\n"
            "2. Identify and name only the clearly visible geometric shapes in each image. Do not assume any shape unless it is clearly present and distinct. If only one shape is visible, say so.\n"
            "3. Compare them: are the same types and number of primitives present in both?\n"
            "4. If any primitives are **missing** in the candidate image (compared to the reference), propose a fix.\n\n"
            "Your fix must include an 'insert_child' action to add each missing shape.\n"
            "- Only insert shapes that clearly appear in the reference but are absent in the candidate.\n"
            "- Use approximate parameters (x, y, size/radius), but stay within a [0, 128] canvas range.\n"
            "- Avoid placing shapes on top of existing ones.\n\n"
            "Use the following format for each fix:\n"
            "  { 'action': 'insert_child', 'target_path': [0], 'new_node': { 'type': '[Circle|Square]', 'x': [x], 'y': [y], '[r|s]': [value] } }\n"
            "  - Use 's' for Square (side length), 'r' for Circle (radius).\n"
            "Return format:\n"
            "explanation: [your reasoning here]\n"
            "edits: [only the list of edits, or an empty list if no edits are needed."
        )

        response = self.vlm_ask_multi([candidate_image, ground_truth_image], question)
        print("[primitive_edit] VLM Response:\n", response)

        try:
            match = re.search(r"\[\s*{.*?}\s*\]", response, re.DOTALL)
            if match:
                try:
                    edits = json.loads(match.group(0).replace("'", '"'))
                except:
                    edits = ast.literal_eval(match.group(0))  # fallback
            else:
                edits = []
        except Exception as e:
            print("[primitive_edit] Failed to parse edit list:", e)
            edits = []

        return edits

    def bounding_box_edit(self, candidate_image, gt_image, image_size):
        """
        Asks the VLM to detect approximate bounding boxes for shapes in both images.
        Normalizes the bounding boxes to [0,1] using image_size: (width, height).

        Returns:
        {
        "candidate": [ { "type": ..., "bbox": [x, y, w, h] (normalized) }, ... ],
        "ground_truth": [ { "type": ..., "bbox": [x, y, w, h] (normalized) }, ... ]
        }
        """
        question = (
            "You are a visual geometry assistant.\n\n"
            "You will be shown two images: the left is the candidate image rendered from a program, "
            "and the right is the ground truth image. Your task is to:\n"
            "1. Detect each geometric shape (e.g., Square, Circle, Triangle, Ellipse) in both images.\n"
            "2. For each shape, return an approximate bounding box: [x, y, width, height] in **pixel values**.\n"
            "3. Group the results by image: `candidate` and `ground_truth`.\n"
            "Important: bounding boxes are not the true shape geometry. They are used only to reason about size and position.\n"
            "Output must be a dictionary like this:\n"
            "{\n"
            "  \"candidate\": [ {\"type\": \"Square\", \"bbox\": [10, 20, 30, 30]}, ... ],\n"
            "  \"ground_truth\": [ {\"type\": \"Circle\", \"bbox\": [15, 25, 28, 28]}, ... ]\n"
            "}"
        )

        response = self.vlm_ask_multi([candidate_image, gt_image], question)
        print("[bounding_box_edit] VLM Response:\n", response)

        # Attempt to parse
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                raw_bboxes = json.loads(match.group(0).replace("'", '"'))
            else:
                raw_bboxes = {"candidate": [], "ground_truth": []}
        except Exception as e:
            print("[bounding_box_edit] Failed to parse output:", e)
            raw_bboxes = {"candidate": [], "ground_truth": []}

        # Normalize bboxes
        W, H = image_size
        def normalize(bbox):
            x, y, w, h = bbox
            return [x / W, y / H, w / W, h / H]

        for key in ["candidate", "ground_truth"]:
            for shape in raw_bboxes.get(key, []):
                shape["bbox"] = normalize(shape["bbox"])

        return raw_bboxes

    def tool_call(self, current_image: Image.Image, gt_image: Image.Image, gt_bbox: tuple[int, int, int], grid_hw: int = 8):
        """
        Compute a CLIP-based perceptual heatmap and similarity score.

        Args:
            current_image (Image.Image): candidate program rendering
            gt_image (Image.Image): ground truth rendering
            gt_bbox (tuple[int, int, int]): (x, y, size) square crop from GT image
            grid_hw (int): grid resolution for patch comparison

        Returns:
            dict with:
                - heatmap: (grid_hw x grid_hw) numpy array of similarities
                - similarity_score: global cosine similarity between whole images
                - overlay: PIL image visualising the heatmap
        """
        assert self.clip_model is not None, "CLIP model is not loaded."

        def clip_embed(img: Image.Image):
            x = self.clip_preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                z = self.clip_model.encode_image(x)
                return z / z.norm(dim=-1, keepdim=True)

        # -- Global similarity
        z_current = clip_embed(current_image)
        z_gt = clip_embed(gt_image)
        similarity_score = float((z_current @ z_gt.T).item())

        # -- Local similarity heatmap
        x_gt, y_gt, sz = gt_bbox
        gt_patch = crop(gt_image, y_gt, x_gt, sz, sz)
        z_patch = clip_embed(gt_patch)

        stride = (current_image.size[0] - sz) // (grid_hw - 1)
        heatmap = np.zeros((grid_hw, grid_hw), dtype=np.float32)

        for gy in range(grid_hw):
            for gx in range(grid_hw):
                y0 = gy * stride
                x0 = gx * stride
                patch = crop(current_image, y0, x0, sz, sz)
                z = clip_embed(patch)
                heatmap[gy, gx] = (z @ z_patch.T).item()

        # Normalize for visualization
        heatmap -= heatmap.min()
        heatmap /= heatmap.max() + 1e-6

        # Overlay
        heat = cv2.resize(heatmap, current_image.size, interpolation=cv2.INTER_NEAREST)
        heat_color = cv2.applyColorMap((255 * (1 - heat)).astype(np.uint8), cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        heat_pil = Image.fromarray(heat_color)
        heat_pil = ImageEnhance.Brightness(heat_pil).enhance(1.4)
        overlay = Image.blend(current_image.convert("RGB"), heat_pil, alpha=0.55)

        return {
            "heatmap": heatmap,
            "similarity_score": similarity_score,
            "overlay": overlay
        }

    def reflexion_phase_1(self, memory, program, current_image, gt_image, max_attempts):
        """
        Align the types of primitives in the candidate and ground-truth images.
        Uses VLM-based reflection to judge whether each proposed edit is valid.
        Even when no edits are proposed, ask the VLM to confirm alignment.
        Stores successful edits in memory and returns the updated program and image.
        """
        attempts = 0

        while attempts < max_attempts:
            proposed_edits = self.primitive_edit(current_image, gt_image)

            # If no edits were proposed, still ask the VLM if types now match
            if not proposed_edits:
                print(f"[Phase 1] No edits proposed at attempt {attempts+1}. Verifying with VLM...")

                reflect_question = (
                    "You are a visual geometry assistant.\n"
                    "Determine if the candidate image contains the same types of geometric primitives "
                    "as the ground-truth image (e.g., Circle, Square, Triangle, Ellipse).\n"
                    "Reply with 'yes' if the primitive types match, otherwise reply with 'no'."
                )

                reflect_response = self.vlm_ask_multi([current_image, gt_image], reflect_question)
                print(f"[Phase 1] Final Reflect Response (attempt {attempts+1}):\n", reflect_response)

                if reflect_response.strip().lower().startswith("yes"):
                    print("[Phase 1] VLM confirms alignment. Done.")
                    return program, current_image
                else:
                    print("[Phase 1] VLM disagrees. Continuing attempts.")
                    attempts += 1
                    continue

            # Otherwise, try applying the first edit
            for edit in proposed_edits:
                updated_program = apply_edit(program, edit)
                program = updated_program
                current_image = render_program(program)
                memory.setdefault("phase1_edits", []).append(edit)
            print(f"Program after edits in attempt {attempts+1}:\n", program)
            new_image = render_program(program)

            # Reflect on whether the edit worked
            reflect_question = (
                "You are a visual geometry assistant.\n"
                "Compare the ground-truth image and the candidate image.\n"
                "Determine whether the **types of geometric primitives** are aligned.\n\n"
                "- Only consider shape types (e.g., Circle, Square).\n"
                "- Ignore position or size.\n"
                "- Reply with 'yes' in judgement if both images contain exactly the same types of primitives (regardless of number or placement).\n"
                "- Reply with 'no' in judgement if any primitive type is present in one image but missing in the other."
                "Return formatted as follows:\n"
                "Explanation: [your reasoning here]\n"
                "Judgment: [yes/no]\n"
            )

            reflect_response = self.vlm_ask_multi([new_image, gt_image], reflect_question)
            print(f"[Phase 1] Reflect Response (attempt {attempts+1}):\n", reflect_response)

            judgment_line = next(
                (line for line in reflect_response.splitlines() if line.lower().startswith("judgment:")), 
                ""
            )
            judgment = judgment_line.split(":")[-1].strip().lower()

            if judgment == "yes":
                current_image = new_image
                print(f"[Phase 1] Accepted edit at attempt {attempts+1}.")
                return program, current_image, memory
            else:
                print(f"[Phase 1] Reflection suggests edit was insufficient at attempt {attempts+1}.")
                attempts += 1
        print("[Phase 1] Max attempts reached. The edit was not successful.")
        return program, current_image, memory

    def reflexion_phase_2(self, memory, program, current_image, gt_image, max_steps=5):
        """
        Refines the size and position of primitives. VLM decides whether to keep or revert each edit.
        """
        image_size = current_image.size
        prev_clip_score = self.tool_call(current_image, gt_image, (0, 0, 64))["similarity_score"]
        prev_boxes = self.bounding_box_edit(current_image, gt_image, image_size)
        prev_program = program
        prev_image = current_image

        memory.setdefault("phase2_edits", [])

        for step in range(max_steps):
            # Ask VLM to propose an edit
            reasoning_prompt = (
                "You are a visual reasoning assistant refining a candidate program to better match a ground-truth image.\n"
                f"The current program is: {program}\n"
                "Primitive types are correct. Improve the **position**, **size**, or **shape** of primitives.\n"
                "Propose exactly one edit in this format:\n"
                "{\"action\": ..., \"target_path\": ..., \"new_node\": {...}}\n"
            )
            response = self.vlm_ask_multi([current_image, gt_image], reasoning_prompt)

            try:
                match = re.search(r"{.*}", response, re.DOTALL)
                edit = json.loads(match.group(0).replace("'", '"')) if match else None
            except Exception as e:
                print("[Phase 2] Failed to parse VLM edit:", e)
                break

            if not edit:
                print("[Phase 2] No edit proposed. Terminating.")
                break

            # Apply and re-render
            memory["phase2_edits"].append(edit)
            updated_program = apply_edit(program, edit)
            updated_image = render_program(updated_program)

            # Re-evaluate
            new_clip_result = self.tool_call(updated_image, gt_image, (0, 0, 64))
            new_clip_score = new_clip_result["similarity_score"]
            new_bboxes = self.bounding_box_edit(updated_image, gt_image, image_size)

            # Let VLM decide whether to keep or revert
            decision_prompt = (
                "You are a visual reasoning agent evaluating whether a program edit improved image alignment.\n"
                f"Previous program:\n{prev_program}\n\n"
                f"Current program:\n{updated_program}\n\n"
                f"Previous CLIP similarity: {prev_clip_score:.4f}\n"
                f"Current CLIP similarity: {new_clip_score:.4f}\n"
                "Both candidate and ground-truth images are provided.\n"
                "Consider the bounding box alignment and semantic similarity.\n"
                "Should we keep the new edit? Reply with only one word: 'keep' or 'revert'."
            )
            decision = self.vlm_ask_multi([prev_image, updated_image, gt_image], decision_prompt).lower().strip()
            print(f"[Phase 2] VLM decision: {decision}")

            if decision.startswith("keep"):
                program = updated_program
                current_image = updated_image
                prev_clip_score = new_clip_score
                prev_program = program
                prev_image = current_image
                prev_boxes = new_bboxes
                print(f"[Phase 2] Step {step+1}: Edit accepted.\n")
            else:
                memory["phase2_edits"].pop()
                print(f"[Phase 2] Step {step+1}: Edit rejected. Reverting.\n")

        return program, current_image
