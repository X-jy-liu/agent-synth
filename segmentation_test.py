import os
import json
import base64
import io
import numpy as np
from PIL import Image, ImageDraw
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def parse_json(json_output: str):
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])
            return json_output.split("```")[0]
    return json_output

def extract_primitives_with_masks(image_path: str, output_dir: str = "segmentation_outputs"):
    # Load and resize image
    im = Image.open(image_path)
    im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
    width, height = im.size

    # Prompt to Gemini
    prompt = """
    You are a vision assistant. Segment all the geometric shapes in the image and return the result strictly in JSON format.

    Each shape must be represented as a JSON object with these three fields:
    - "label": a descriptive string, e.g. "red circle" or "blue square"
    - "box_2d": a list of 4 numbers in the format [ymin, xmin, ymax, xmax], normalized to the range 0–1000
    - "mask": a 2D list of 0s and 1s representing a binary segmentation mask, where 1 indicates foreground pixels

    Return a JSON object with a top-level key called "segmentation_masks", which maps to a list of shape entries.

    ⚠️ Output **only valid JSON** — no markdown, no explanations, no comments, no text before or after the JSON.

    Example output:
    {
    "segmentation_masks": [
        {
        "label": "blue square",
        "box_2d": [100, 150, 300, 350],
        "mask": [
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            ...
        ]
        },
        ...
    ]
    }
    """

    # Call Gemini
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        contents=[prompt, im],
        generation_config={"response_mime_type": "application/json"}
    )

    # Parse model response
    try:
        parsed = parse_json(response.text)
        items = json.loads(parsed)
    except Exception as e:
        print("Failed to parse response:", e)
        print(response.text)
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"items: {items}")
    items = items["segmentation_masks"]  # unwrap the actual list
    for i, item in enumerate(items):
        if isinstance(item, str):
            print(f"Warning: item[{i}] is a string, not a dict: {item}")
            continue  # Skip or handle as needed
        label = item.get("label", f"object_{i}")
        box = item["box_2d"]
        ymin, xmin, ymax, xmax = box
        x0 = int(xmin / 1000 * width)
        y0 = int(ymin / 1000 * height)
        x1 = int(xmax / 1000 * width)
        y1 = int(ymax / 1000 * height)

        if y0 >= y1 or x0 >= x1:
            continue

        # Handle mask
        mask_data_uri = item.get("mask", "")
        if not mask_data_uri.startswith("data:image/png;base64,"):
            continue
        mask_base64 = mask_data_uri.split(",")[1]
        mask_data = base64.b64decode(mask_base64)
        mask_img = Image.open(io.BytesIO(mask_data)).resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)
        mask_array = np.array(mask_img)

        # Overlay drawing
        overlay = Image.new('RGBA', im.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        color = (255, 255, 255, 180)

        for y in range(y0, y1):
            for x in range(x0, x1):
                if mask_array[y - y0, x - x0] > 128:
                    overlay_draw.point((x, y), fill=color)

        # Save outputs
        mask_filename = f"{label}_{i}_mask.png"
        overlay_filename = f"{label}_{i}_overlay.png"
        mask_img.save(os.path.join(output_dir, mask_filename))
        composite = Image.alpha_composite(im.convert("RGBA"), overlay)
        composite.save(os.path.join(output_dir, overlay_filename))
        print(f"Saved mask and overlay for {label} to {output_dir}")

# Example usage
if __name__ == "__main__":
    extract_primitives_with_masks("/home/jingyang/agent-synth/tool_call_with_clip_embedding/images/gt_image.png")