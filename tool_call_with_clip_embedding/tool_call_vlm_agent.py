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

class VLMEditAgent:
    def __init__(self,
                vlm_type: str,
                device: str = None,
                clip_model_name: str = "ViT-B/32",
                output_dir: str = "outputs",
                openai_api_key: str = None,
                anthropic_api_key: str = None,
                gemini_api_key: str = None):
        
        self.vlm_type = vlm_type.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Store keys
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.gemini_api_key = gemini_api_key

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

    def primitive_edit(self, candidate_image, gt_image):
        """
        Compare the candidate image and ground truth image.
        If any primitives are missing in the candidate, propose an insert edit.
        
        Returns:
            {
                "need_edit": True or False,
                "edit": {...}  # Only if True
            }
        """
        question = (
            "You are a vision-based program repair agent.\n"
            "Task:\n"
            "1. Look at both images provided.\n"
            "2. Identify all geometric primitives in each (e.g., Square, Circle, Triangle, Ellipse).\n"
            "3. Compare them: are the same types of primitives present in both?\n"
            "4. If any primitives are **missing** in the candidate image, propose a fix.\n\n"
            "Your fix should include an 'insert' action to add the missing shape(s).\n"
            "Use approximate positions and sizes — they do not need to be precise.\n"
            "Format your answer as a list of edits using this format:\n"
            "  { 'action': 'insert_child', 'target_path': [0], 'new_node': { 'type': 'Circle', 'x': 30, 'y': 30, 'r': 10 } }\n"
            "Return only the list of edits or an empty list if no edits are needed."
        )

        response = self.vlm_ask_multi([candidate_image, gt_image], question)
        print("[primitive_edit] VLM Response:\n", response)

        # Try to extract JSON list of edits
        try:
            # Extract list of JSON-like dicts using regex fallback
            match = re.search(r"\[\s*{.*?}\s*\]", response, re.DOTALL)
            if match:
                edits = json.loads(match.group(0).replace("'", '"'))
            else:
                edits = []
        except Exception as e:
            print("[primitive_edit] Failed to parse edit list:", e)
            edits = []

        if edits:
            return {"need_edit": True, "edit": edits[0]}  # Only apply first edit
        else:
            return {"need_edit": False}

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

    def reflexion_phase_1(self, memory, program, current_image, gt_image):
        """
        Align the types of primitives in the candidate and ground-truth images.
        Stores successful edits in memory and returns the updated program and image.
        """
        max_attempts = 5
        attempts = 0
        failed_edits = []

        while attempts < max_attempts:
            result = self.primitive_edit(current_image, gt_image)

            if not result["need_edit"]:
                print("[Phase 1] Primitive types aligned.")
                return program, current_image

            edit = result["edit"]
            memory.setdefault("phase1_edits", []).append(edit)
            print(f"[Phase 1] Attempt {attempts+1}: Proposed edit -> {edit}")

            # Apply the edit to the program (user must define this)
            program = apply_edit(program, edit)

            # Re-render the updated program into a new image (user must define this)
            current_image = render_program(program)

            # Optional: verify if primitive types now match; if not, continue
            result_check = self.primitive_edit(current_image, gt_image)
            if not result_check["need_edit"]:
                print("[Phase 1] Alignment achieved after edit.")
                return program, current_image

            failed_edits.append((edit, result_check))
            attempts += 1

        print("[Phase 1] Max attempts reached. Returning last candidate.")
        return program, current_image


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
