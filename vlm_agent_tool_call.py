import openai
from io import BytesIO
from PIL import Image
import base64
import json
import re
from utils import crop_difference_region


def image_to_base64(img: Image.Image):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def crop_object_region(program_code, target_path, image, padding=0):
    """
    Crop a region around the object at target_path in the program.
    Returns both the cropped image and the (xmin, ymin) offset.
    """
    node = program_code
    for idx in target_path:
        node = node["children"][idx] if "children" in node else node[idx]

    x = node.get("x")
    y = node.get("y")
    r = node.get("r", node.get("size", 20))

    xmin = max(0, x - r - padding)
    ymin = max(0, y - r - padding)
    xmax = x + r + padding
    ymax = y + r + padding

    cropped = image.crop((xmin, ymin, xmax, ymax))
    return cropped, (xmin, ymin)


def vlm_zoom_select(current_img, target_img, diff_img, program_code, prev_iou, current_iou):
    current_b64 = image_to_base64(current_img)
    target_b64 = image_to_base64(target_img)
    diff_b64 = image_to_base64(diff_img)

    improved = current_iou > prev_iou
    feedback = "Good progress ✅ IoU improved." if improved else "Failed ⚠️ — IoU dropped or unchanged."

    prompt = f"""
You are improving a shape-based image synthesis program.

Previous IoU: {prev_iou:.3f}
Current IoU: {current_iou:.3f}
Feedback: {feedback}

You are working object-by-object. Your goal is to identify the single most problematic primitive in the current program — the one that contributes most to the visual difference from the target.

To do this:
1. Look at the **highlighted diff image**.
2. Inspect the current and target images.
3. Choose the object to zoom into using its `target_path` from the program below.

You are NOT proposing an edit yet — just calling `zoom_into_object` with one `target_path`.

Here is the current program:
{json.dumps(program_code, indent=2)}
"""

    messages = [
        {"role": "system", "content": "You are a VLM agent that identifies which object to inspect based on visual differences."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_b64}", "detail": "low"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{target_b64}", "detail": "low"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{diff_b64}", "detail": "high"}},
        ]}
    ]

    tool_spec = [
        {
            "type": "function",
            "function": {
                "name": "zoom_into_object",
                "description": "Zoom into the region around a specific primitive for detailed inspection.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_path": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Path to the object to zoom into (e.g., [0] or [2, 1])"
                        }
                    },
                    "required": ["target_path"]
                }
            }
        }
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tool_spec,
        tool_choice={"type": "function", "function": {"name": "zoom_into_object"}},
        temperature=0.2,
        max_tokens=300
    )

    response_message = response.choices[0].message

    if response_message.tool_calls:
        tool_call = response_message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        return arguments["target_path"]
    else:
        print("⚠️ No zoom tool call made. Agent failed to pick an object.")
        raise ValueError("No zoom_into_object tool call made.")


def vlm_propose_edit(current_crop, target_crop, diff_crop, program_code, target_path, memory=[]):
    current_b64 = image_to_base64(current_crop)
    target_b64 = image_to_base64(target_crop)
    diff_b64 = image_to_base64(diff_crop)

    memory_str = "\n\n".join([
        f"Step {i+1}:\nThought: {m['thought']}\nEdit: {json.dumps(m['edit'])}\nIoU: {m['prev_iou']} → {m['current_iou']}"
        for i, m in enumerate(memory)
    ]) if memory else "No memory yet."

    connection_block = f"""
You are now focused on one object only:

- The zoomed-in images below show the object at `target_path = {target_path}`.
- These zooms are cropped from the full images.
- You must propose ONE atomic edit **on the full program**, not the crop.

Always use `target_path = {target_path}` and do NOT guess another path.
"""

    prompt = f"""
You are proposing an atomic edit to improve a shape-based program.

Last 3 steps of memory:
{memory_str}

{connection_block}

Here is the current program:
{json.dumps(program_code, indent=2)}

Step-by-step:
1. Think heuristically — what might still be wrong?
2. Output your reasoning in the `thought`.
3. Call `propose_edit`.
"""

    messages = [
        {"role": "system", "content": "You are a VLM agent that proposes edits to shape programs one object at a time."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_b64}", "detail": "high"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{target_b64}", "detail": "high"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{diff_b64}", "detail": "high"}},
        ]}
    ]

    tool_spec = [{
        "type": "function",
        "function": {
            "name": "propose_edit",
            "description": "Suggest a single atomic edit to improve the shape program.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "action": {"type": "string", "enum": ["insert_child", "modify_param"]},
                    "target_path": {
                        "type": "array",
                        "items": {"type": "integer"}
                    },
                    "param": {
                        "type": "string",
                        "enum": ["x", "y", "r", "size"]
                    },
                    "value": {"type": "integer"},
                    "new_node": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["Circle", "Square"]},
                            "x": {"type": "integer"},
                            "y": {"type": "integer"},
                            "r": {"type": "integer"},
                            "size": {"type": "integer"}
                        }
                    }
                },
                "required": ["thought", "action", "target_path"]
            }
        }
    }]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tool_spec,
        tool_choice={"type": "function", "function": {"name": "propose_edit"}},
        temperature=0.2,
        max_tokens=500
    )

    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    thought = args.pop("thought", "No thought provided")
    return thought, args