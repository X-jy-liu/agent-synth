import re
import json
import logging


def parse_answer(response: str):
    if not response or not isinstance(response, str):
        return ""
    
    # Use DOTALL flag to handle multiline content
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL | re.IGNORECASE)
    
    if match:
        content = match.group(1)
        # Convert to single line: replace newlines/tabs with spaces, compress multiple spaces
        single_line = re.sub(r'\s+', ' ', content).strip()
        return single_line
    
    # Fallback: return original response as single line
    return re.sub(r'\s+', ' ', response).strip()


def parse_answer_json(llm_response: str) -> dict:
    """
    Extract and parse JSON from <answer>...</answer> tags.
    
    Args:
        llm_response (str): The full LLM response containing <answer> tags
        
    Returns:
        dict: Parsed JSON as a dictionary
        
    Raises:
        ValueError: If no answer tags found or JSON is invalid
    """
    # Extract content between <answer> and </answer>
    match = re.search(r'<answer>(.*?)</answer>', llm_response, re.DOTALL)
    
    if not match:
        raise ValueError("No <answer>...</answer> tags found in response")
    
    answer_content = match.group(1).strip()
    
    try:
        # Parse the JSON string into a dictionary
        return json.loads(answer_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in answer tags: {e}")
    

def format_message(sys_prompt=None, user_prompt=None):
    message = []
    if sys_prompt:
        message.append({"role": "system", "content": sys_prompt})
    if user_prompt:
        message.append({"role": "user", "content": user_prompt})
        
    return message

def parse_between_tags(text: str, tag_name: str, default: str = "") -> str:
    """
    Extract content between XML-style tags from text.
    
    Args:
        text: The text to search in
        tag_name: The name of the tag (without < >)
        default: Default value to return if tag not found
    
    Returns:
        str: Content between the tags, or default if not found
    """
    try:
        # Create pattern for both self-closing and regular tags
        pattern = f'<{tag_name}[^>]*>(.*?)</{tag_name}>'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            content = match.group(1).strip()
            return content
        
        # Try self-closing tag pattern as fallback
        self_closing_pattern = f'<{tag_name}[^>]*/>'
        if re.search(self_closing_pattern, text, re.IGNORECASE):
            return ""  # Empty content for self-closing tags
            
        return default
        
    except Exception as e:
        logging.warning(f"Error parsing tag '{tag_name}': {e}")
        return default


if __name__ == "__main__":
    response = '''
 <think>
1. Current Visual Analysis  
• Overall layout = horizontal split (gap 2).  
  – Left side (v-stack, gap 1): tall purple bar at (+0,+1) above a small blue dot at (+0,-2).  
  – Right side (v-stack, gap 1):  
     • Top row (h-stack, gap 2): slim black oval at (+1,+0) left of a 9×9 green circle at (-3,+1).  
     • Bottom row (h-stack, gap 2): 5×1 green bar at (-3,+1) left of a vertical red stub.  

2. VLM Intent Analysis  
• Green bar: should sit a little higher and slightly right so it centres beneath the oval/circle pair.  
• Green circle: needs to rise so its top aligns roughly with the black oval’s mid-height and look a bit larger.  
• Purple bar: should drop two units so its base aligns with the rest of the composition.  

3. Strategic Options  
a) Re-arrange groups – unnecessary; only fine tweaks.  
b) Directly edit the three affected primitives’ offsets/sizes – simplest and keeps structure intact.  
c) For “larger” green circle, true scaling past 9 isn’t allowed, so an elegant proxy is to thicken its stroke (visually increasing its footprint) while keeping width/height within limits.

4. Optimal Solution  
Choose option (b) with a stroke-width tweak for the circle:  
• Purple bar y +1 → -1  (MOVE_DOWN ×2).  
• Green bar  (-3,+1) → (-2,+3)  (UP ×2, RIGHT ×1).  
• Green circle y +1 → +2 (UP ×1) and stroke 1 → 2 to appear larger.

5. Implementation Plan  
Edit only the three primitives; preserve every Arrange and gap value.  
</think>

<answer>
(Arrange h
  (Arrange v
    (Rectangle 2 7 purple red 1 +0 -1)
    (Ellipse 1 1 blue none 0 +0 -2)
    1)
  (Arrange v
    (Arrange h
      (Ellipse 2 6 black none 0 +1 +0)
      (Ellipse 9 9 green orange 2 -3 +2)
      2)
    (Arrange h
      (Rectangle 5 1 green none 0 -2 +3)
      (Arrange v
        (Rectangle 1 1 red none 0 +0 +1)
        (Ellipse 2 2 red none 0 +0 +0)
        1)
      2)
    1)
  2)
</answer>
'''

    print(parse_answer(response))