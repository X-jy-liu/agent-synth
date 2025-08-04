import os
import base64
import mimetypes
from openai import OpenAI
from typing import List, Dict, Optional

# instantiate once
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_llm(
    messages: List[Dict[str, str]],
    model_name: str = "gpt-4o",
    temperature: float = 1,
    max_tokens: int = 2000,
) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def local_image_to_data_url(image_path: str) -> str:
    """
    Convert a local image file to a base64-encoded data URL.
    """
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "application/octet-stream"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def call_vlm(
    messages: List[Dict[str, str]],
    image_paths: List[str],
    model_name: str = "gpt-4o",
    temperature: float = 1,
    max_tokens: int = 2000,
) -> str:
    """
    Embed multiple local images + text into a vision-enabled chat model.

    :param messages: same format as call_llm; last user message is the prompt
    :param image_paths: list of paths to your image files (png/jpg/etc)
    :param model_name: e.g. "gpt-4-vision-preview" or any vision-capable model
    :param temperature: sampling temperature
    :param max_tokens: max tokens in reply
    :return: assistant's descriptive reply
    """
    # convert all images to data URLs
    data_urls = [local_image_to_data_url(path) for path in image_paths]

    # locate last user message
    last_user_idx = max(i for i, m in enumerate(messages) if m["role"] == "user")

    # build new messages list with images embedded
    new_msgs: List[Dict] = []
    for i, m in enumerate(messages):
        if i == last_user_idx:
            # Start with the text content
            content = [{"type": "text", "text": m["content"]}]
            
            # Add all images
            for data_url in data_urls:
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": data_url}
                })
            
            new_msgs.append({
                "role": "user",
                "content": content
            })
        else:
            new_msgs.append(m)

    resp = client.chat.completions.create(
        model=model_name,
        messages=new_msgs,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def call_vlm_flexible(
    messages: List[Dict[str, str]],
    image_paths: List[str] = None,
    image_path: str = None,  # backward compatibility
    model_name: str = "gpt-4o",
    temperature: float = 0.3,
    max_tokens: int = 500,
) -> str:
    """
    Embed local images + text into a vision-enabled chat model.
    Supports both single image (backward compatibility) and multiple images.

    :param messages: same format as call_llm; last user message is the prompt
    :param image_paths: list of paths to your image files (preferred)
    :param image_path: single path for backward compatibility
    :param model_name: e.g. "gpt-4-vision-preview" or any vision-capable model
    :param temperature: sampling temperature
    :param max_tokens: max tokens in reply
    :return: assistant's descriptive reply
    """
    # Handle backward compatibility and argument validation
    if image_paths is None and image_path is None:
        raise ValueError("Must provide either image_paths (list) or image_path (single)")
    
    if image_paths is None:
        image_paths = [image_path]
    elif image_path is not None:
        # Both provided, prefer image_paths but warn
        print("Warning: Both image_paths and image_path provided. Using image_paths.")
    
    # convert all images to data URLs
    data_urls = [local_image_to_data_url(path) for path in image_paths]

    # locate last user message
    last_user_idx = max(i for i, m in enumerate(messages) if m["role"] == "user")

    # build new messages list with images embedded
    new_msgs: List[Dict] = []
    for i, m in enumerate(messages):
        if i == last_user_idx:
            # Start with the text content
            content = [{"type": "text", "text": m["content"]}]
            
            # Add all images
            for data_url in data_urls:
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": data_url}
                })
            
            new_msgs.append({
                "role": "user",
                "content": content
            })
        else:
            new_msgs.append(m)

    resp = client.chat.completions.create(
        model=model_name,
        messages=new_msgs,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# Example usage:
if __name__ == "__main__":
    # Multiple images
    messages = [
        {"role": "user", "content": "Compare these two images and describe the differences."}
    ]
    
    result = call_vlm(
        messages=messages,
        image_paths=["target.png", "current.png"]
    )
    print(result)
    
    # Single image (using the flexible version)
    result2 = call_vlm_flexible(
        messages=[{"role": "user", "content": "Describe this image."}],
        image_path="single_image.png"  # backward compatibility
    )
    print(result2)


if __name__ == "__main__":
    # 1) Text-only
    reply = call_llm(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": "Explain quantum entanglement in simple terms."}
        ],
        model_name="gpt-4o",
        temperature=0.5,
        max_tokens=200
    )
    print(reply)

    # 2) Image + text
    reply = call_vlm(
        messages=[
            {"role": "system", "content": "You are an image-expert assistant."},
            {"role": "user",   "content": "What’s happening in this photo?"}
        ],
        image_path="/Users/zhanghantao/Desktop/LLM_Spatial_Reasoning/LLM_3D/scene_understanding3D/images/exp3D/task_1/task_1_3d_view_01.png",
        model_name="gpt-4o",
        temperature=0.3,
        max_tokens=300
    )
    print(reply)
