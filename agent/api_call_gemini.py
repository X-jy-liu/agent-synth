import os
from google import genai
from google.genai import types
from PIL import Image
from typing import List, Dict

# Configure Gemini API
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def format_for_gemini(messages: List[Dict[str, str]]):
    gemini_messages = []
    sys_instruction = ""
    for msg in messages:
        if msg['role'] == "user":
            gemini_messages.append(msg['content'])
        elif msg['role'] == "system":
            sys_instruction += msg['content']
    return gemini_messages, sys_instruction


def call_llm(
    messages: List[Dict[str, str]],
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.3,
    max_tokens: int = 5000,
) -> str:
    
    gemini_messages, sys_instruction = format_for_gemini(messages=messages)
    
    response = client.models.generate_content(
        model=model_name,
        contents=gemini_messages,
        config=types.GenerateContentConfig(
            system_instruction=sys_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens)
    )
    # print(response.text)
    return response.text


def call_vlm(
    messages: List[Dict[str, str]],
    image_path: str,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.3,
    max_tokens: int = 5000,
) -> str:
    
    image = Image.open(image_path)
    
    gemini_messages, sys_instruction = format_for_gemini(messages=messages)
    gemini_messages.append(image)
    
    response = client.models.generate_content(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=sys_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens),
        contents=gemini_messages
    )
    
    return response.text


# Example usage
if __name__ == "__main__":
    # 1) Text-only
    reply = call_llm(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": "Explain quantum entanglement in simple terms."}
        ],
        model_name="gemini-2.5-pro",
        temperature=0.3,
        max_tokens=2000
    )
    print(reply)

    # 2) Image + text
    reply = call_vlm(
        messages=[
            {"role": "system", "content": "You are an image-expert assistant."},
            {"role": "user",   "content": "What’s happening in this photo?"}
        ],
        image_path="/Users/zhanghantao/Desktop/LLM_Spatial_Reasoning/LLM_3D/scene_understanding3D/images/exp3D/task_1/task_1_3d_view_01.png",
        model_name="gemini-2.5-pro",
        temperature=0.3,
        max_tokens=1000
    )
    print(reply)