import os
import json
from PIL import Image, ImageDraw
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load image
image_path = "/home/jingyang/agent-synth/tool_call_with_clip_embedding/images/gt_image.png"
image = Image.open(image_path)
width, height = image.size

# Prepare prompt
prompt = (
    "Detect all of the primitives in the image. "
    "The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0–1000. "
    "Return the result as a list of JSON objects."
)

# Load vision model
model = genai.GenerativeModel("gemini-1.5-flash")  # or "gemini-pro-vision"

# Request response as JSON
response = model.generate_content(
    contents=[prompt, image],
    generation_config={"response_mime_type": "application/json"}
)

# Parse response
try:
    bounding_boxes = json.loads(response.text)
except json.JSONDecodeError:
    print("Failed to parse response as JSON:")
    print(response.text)
    exit(1)

# Convert normalized coordinates to absolute pixel values
converted_bounding_boxes = []
for box in bounding_boxes:
    ymin, xmin, ymax, xmax = box["box_2d"]
    abs_x1 = int(xmin / 1000 * width)
    abs_y1 = int(ymin / 1000 * height)
    abs_x2 = int(xmax / 1000 * width)
    abs_y2 = int(ymax / 1000 * height)
    converted_bounding_boxes.append([abs_x1, abs_y1, abs_x2, abs_y2])

# Output
print("Image size:", width, height)
print("Bounding boxes:", converted_bounding_boxes)

# Draw boxes on image
draw = ImageDraw.Draw(image)
for bbox in converted_bounding_boxes:
    draw.rectangle(bbox, outline="red", width=3)

# Save or show image
output_path = "/home/jingyang/agent-synth/tool_call_with_clip_embedding/images/annotated_image.png"
image.save(output_path)
print("Annotated image saved to:", output_path)