import base64
from io import BytesIO
from PIL import Image

# Optional imports depending on backend
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# GPT-4V: accepts one image, so we concatenate two images side-by-side.
def ask_openai_multi(images: list[Image.Image], question: str, api_key: str) -> str:
    if openai is None:
        raise ImportError("openai package is not installed.")

    if len(images) != 2:
        raise ValueError("GPT-4V currently only supports one image. Concatenate if needed.")

    # Concatenate side-by-side
    w, h = images[0].size
    canvas = Image.new("RGB", (w * 2, h))
    canvas.paste(images[0], (0, 0))
    canvas.paste(images[1], (w, 0))

    openai.api_key = api_key
    buffered = BytesIO()
    canvas.save(buffered, format="PNG")
    b64_image = base64.b64encode(buffered.getvalue()).decode()

    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question + "\n(Left: candidate image, Right: ground truth image)"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ]
        }],
        max_tokens=1000
    )
    return response['choices'][0]['message']['content']

# Claude 3: supports true multi-image input via base64.
def ask_claude_multi(images: list[Image.Image], question: str, api_key: str) -> str:
    if anthropic is None:
        raise ImportError("anthropic package is not installed.")

    client = anthropic.Anthropic(api_key=api_key)
    media = []
    for img in images:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        b64 = base64.b64encode(buffered.getvalue()).decode()
        media.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}})

    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": question}] + media
        }]
    )
    return response.content[0].text

# Gemini 1.5: supports multi-image input directly.
def ask_gemini_multi(images: list[Image.Image], question: str, api_key: str) -> str:
    if genai is None:
        raise ImportError("google.generativeai package is not installed.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro-vision")
    inputs = [question] + images
    response = model.generate_content(inputs, stream=False)
    return response.text + "\n[model: Gemini 1.5]"
