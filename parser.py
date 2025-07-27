import re


def parse_answer_expression(response: str) -> str:
    
    # Use regex to find content between <answer> and </answer> tags
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, response, re.DOTALL)
    
    if match:
        # Extract the expression and strip whitespace
        expression = match.group(1).strip()
        return expression
    else:
        print("Warning: No <answer> tags found in response")
        return None