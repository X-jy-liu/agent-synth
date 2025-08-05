
LLM_grammar_prompt = """
You are a tinySVG expression generator. Produce nested expressions using exactly the grammar and constraints below. Do not add commentary or invent new primitives, colors, formats, or token styles unless explicitly asked.

Primitives & collections:
- Shapes: only two shape primitives: Rectangle and Ellipse.
  Syntax:
    (Rectangle W H FILL_COLOR STROKE_COLOR STROKE_WIDTH OX OY)
    (Ellipse   W H FILL_COLOR STROKE_COLOR STROKE_WIDTH OX OY)
  * W, H, and STROKE_WIDTH are integer scales in the range 0 to 9 inclusive.
  * FILL_COLOR and STROKE_COLOR must be one of: red, green, blue, yellow, purple, orange, black, white, none. (Lowercase; "none" means no fill or no stroke.)
  * OX and OY are signed integer offsets along x and y, each in range -9 to 9, and must include an explicit sign (e.g., +0, -3, +9).

Composition:
- Arrange operator: layout two subexpressions.
  Syntax:
    (Arrange D A B G)
  * D is direction: 'v' or 'h'. 'v' means stack A above B; 'h' means place A to the left of B.
  * A and B are subexpressions (each can be a Rectangle, Ellipse, or another Arrange).
  * G is the gap: an integer 0 to 9 specifying spacing in the layout direction.

Constraints summary:
- Scale values (width, height, stroke width, gap): 0..9 (digits).
- Offsets OX, OY: signed integers from -9 to 9, with leading + or -.
- Colors: from the fixed set [red, green, blue, yellow, purple, orange, black, white, none].
- Directions: 'v' or 'h' only.
- Nesting is allowed arbitrarily via Arrange.

Output requirements:
- Emit only valid expressions conforming to this grammar; no extra explanation unless explicitly requested.
- Tokens must follow the exact formatting (spaces separating items, parentheses, signs on offsets).
- Respect all ranges and allowed values.

Example:
(Arrange v (Arrange h (Ellipse 3 3 blue none 0 +0 +0) (Rectangle 1 4 yellow black 1 +1 -1) 3) (Arrange h (Rectangle 4 4 orange black 2 +0 +0) (Ellipse 6 2 yellow red 2 +2 -1) 4) 6)
"""


VLM_scene_description_prompt = """
You are given an image composed of simple shape primitives. Your task is to describe each primitive in JSON only, adhering exactly to the schema and rules below. Do not add any prose outside the JSON.

Schema:
{
  "primitives": [
    {
      "id": "<unique deterministic id>",
      "features": {
        "shape": "rect" | "ellipse",
        "fill_color": "<one of red, blue, green, orange, purple, white, black, gray>",
        "stroke_color": "<one of red, blue, green, orange, purple, white, black, gray, none>",
        "pos_bin": "TL" | "TC" | "TR" | "CL" | "C" | "CR" | "BL" | "BC" | "BR"
      },
      "description": "<In-detailed phrase describing its size, shape, and position relative to other primitives>"
    }
  ],
  "bg_color": "<background color name from the same color set except 'none'>"
}

Rules:
1. **ID construction:** Must be simple, deterministic, and unique. Compose it from its features: 
   format: `<shape>-<fill_color>-<stroke_color>-<pos_bin>[-N]`
   If multiple primitives would otherwise collide, append `-1`, `-2`, etc., to make them unique. Example: `ellipse-blue-black-TC-1`.
2. **pos_bin definition:** Divide the image into a 3x3 grid:
   - Rows: Top (T), Center (C), Bottom (B)
   - Columns: Left (L), Center (C), Right (R)
   Combine to get: TL, TC, TR, CL, C, CR, BL, BC, BR. 
   Assign each primitive to the bin corresponding to where its visual center lies.
3. **description:** Provide a in-detailed natural-language description about the relative position, size, shape comparing this primitive to other primitive(s). Examples: 
   - "to the left of red rectangle"
   - "above the green ellipse"
   - "slightly below and overlaps the large blue ellipse on top"
   - "slightly bigger than the red rectangle"
   - "flat rectangle"
   - "thin ellipse and smaller than the blue ellipse".
   Do not use raw coordinates.
4. **Color values:** Use only the allowed color names, lowercase. If a primitive has no stroke, use `"none"` for `stroke_color`.
5. **Output constraints:** 
   - Output valid JSON inside <answer> ... </answer>.
   - Ensure all `id` fields are unique.
   - Order of primitives does not matter.

Example:
<answer>
{
  "primitives": [
    {
      "id": "rect-yellow-black-TL",
      "features": {
        "shape": "rect",
        "fill_color": "yellow",
        "stroke_color": "black",
        "pos_bin": "TL"
      },
      "description": "above the green ellipse and bigger than the green ellipse"
    },
    {
      "id": "ellipse-green-none-CL",
      "features": {
        "shape": "ellipse",
        "fill_color": "green",
        "stroke_color": "none",
        "pos_bin": "CL"
      },
      "description": "to the right of yellow rectangle, very flat"
    }
  ],
  "bg_color": "white"
}
</answer>
"""


LLM_program_synthesis_prompt = """
In addition to the tinySVG grammar explained earlier, you will be given a VLM scene description in JSON.

<<VLM_DESCRIPTION>>
{vlm_description}
<</VLM_DESCRIPTION>>

Explanation of the description fields:
- Each primitive entry contains:
  * "id": a unique deterministic identifier (for bookkeeping; not used directly in the grammar).
  * "features":
      - "shape": either "rect" or "ellipse" — determines whether to use (Rectangle ...) or (Ellipse ...).
      - "fill_color": the interior color of the shape.
      - "stroke_color": the outline color; use "none" if there should be no stroke.
      - "pos_bin": one of TL, TC, TR, CL, C, CR, BL, BC, BR — a coarse spatial bin from a 3x3 grid (Top/Center/Bottom × Left/Center/Right) indicating approximate placement.
  * "relative_position": a short natural-language phrase describing its position relative to other primitives (e.g., "to the left of red rectangle", "above the green ellipse", "isolated in C").
- "bg_color": the background color of the whole scene.

Your task: based on that description, produce a single tinySVG expression (using only the grammar: Rectangle, Ellipse, Arrange with directions 'v'/'h', gaps 0–9, and signed offsets -9..9) that attempts to realize the described scene. Do not include any extra explanation outside the tags.

Output format:
<think>...</think> — a brief summary of how you interpreted the VLM description: what each primitive is, how pos_bin and relative_position informed their layout or grouping, and any key decisions (e.g., which primitives you arranged together and why).  
<answer>...</answer> — one valid tinySVG expression implementing the scene.

Example:
<think>Placed a blue ellipse in the top-left and a yellow rectangle to its right, so I arranged them horizontally with gap 3.</think>
<answer>(Arrange h (Ellipse 4 4 blue none 1 -2 +2) (Rectangle 3 5 yellow black 1 +2 +2) 3)</answer>
"""


VLM_edits_sys = """
You are analyzing two SVG graphics images to identify differences and suggest corrections to guide LLM to modify the current image step by step to match the target.

**Target Image (First Image)**: The desired final result
**Current Image (Second Image)**: The current state that needs modification

You should focus on 1-3 shapes that need major adjustments.

## Analysis Task

1. **Shape-by-Shape Comparison**: 
   - Identify each shape in both images by type (rectangle, circle, ellipse) and visual properties
   - Match corresponding shapes between target and current images
   - Note any missing shapes in current image or extra shapes that shouldn't be there

2. **Detailed Difference Analysis**:
   For each shape, compare:
   - **Position**: Where is it located? (use qualitative relative descriptions)
   - **Size**: How big is it?
   - **Color**: Fill color and stroke color
   - **Orientation**: Any rotation or angle differences
   - **Visibility**: Is the shape present in both images?

3. **Spatial Relationships**:
   - How do shapes relate to each other spatially?
   - Are there alignment issues between target and current?
   - Note any overlapping or spacing problems

## Modification Suggestions

For all descriptions or modification suggestions, only use qualitative and relative descriptions, focusing on the relative size or positions to other shapes or the whole canvas.
E.g. The width should be around one half of the canvas; the size should be doubled; the blue rectangle should just touch the red circle on its left.


**Be specific about:**
- Which shape you're referring to (e.g., "the blue rectangle in the top-left", "the small red circle")
- Directional movements (left/right, up/down)
- Size changes (bigger/smaller)
- Exact color names when possible
"""


VLM_edits_user = """
Identify up to 3 primitives in the current image that needs changes to match the target image.

How to change each primitive to match the TARGET image in terms of position, scale, fill color, and stroke color and width?

Available discrete actions:
MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN,
SCALE_UP_X, SCALE_DOWN_X, SCALE_UP_Y, SCALE_DOWN_Y,
FILL_<color>, STROKE_<color>, STROKE_WIDTH_<+/->

Color palette: [red, blue, green, orange, purple, white, black, gray]

Provide your response in this format:
<think>
Analyze the differences between the target and current primitive. What needs to change in terms of position, size, and colors?
</think>

<answer>
{
    "short primitive description": "action explanation",

}
</answer>

Example:
<answer>
{
    "blue ellipse that is above a orange rectangle": "it should MOVE_LEFT so that it sits on the top left of the orange rect",
    "yellow circle that is next to the blue ellipse": "it should SCALE_DOWN_X so that it forms a thin ellipse rather than a circle"
}
</answer>
Guidelines:
- In the primitive description, give a short natural language description that helps user to identify the primitive
- Choose 1-4 actions that will make the current primitive better match the target
- Action explanation should be explaining how these discrete actions should be implemented and also the expected effects of this action in details
- You are encouraged to use relational languages to describe your actions by comparing a primitive with surrounding primitives on the canvas. E.g. The ellipse should be adjusted to have the same height as the rectangle on its left
- Prioritize the most visually impactful changes first
- Use minimal actions needed
"""


VLM_edits_user_2 = """
Looking at the target image (first) and current generated image (second), suggest modifications to better align the current image with the target.
"""


VLM_edits_with_feedback_prompt = """
Looking at the target image (first) and current generated image (second), suggest modifications to better align the current image with the target.

IMPORTANT FEEDBACK: The previous suggestions below did NOT lead to better alignment with the target image:
{previous_suggestions}

Please provide NEW and DIFFERENT suggestions that:
1. Address different aspects than the failed suggestions
2. Take a different approach to the alignment problem
3. Consider alternative modifications that might be more effective
4. Focus on the most critical differences between the images

Avoid repeating the same types of suggestions that failed previously. Think about alternative strategies for improvement.
"""


LLM_EXPRESSION_MODIFIER_PROMPT = """
You are tasked with modifying a tinySVG expression based on VLM-suggested actions. Your goal is NOT to blindly apply the actions, but to understand the INTENT behind them and achieve the desired visual effect through optimal expression design.

CURRENT EXPRESSION:
{current_expression}

CURRENT VLM ACTIONS:
{current_actions}

tinySVG Grammar Reminder:
- (Rectangle width height fill_color stroke_color stroke_width x_offset y_offset)
- (Ellipse width height fill_color stroke_color stroke_width x_offset y_offset)
- (Arrange direction child1 child2 ... gap)

Nested Structure Understanding:
- Primitives (Rectangle/Ellipse) are LEAF nodes - they cannot contain other elements
- Arrange operations are CONTAINER nodes - they organize and position their children
- Deep nesting is possible: (Arrange h (Arrange v rect1 rect2 gap1) ellipse1 gap2)
- Each Arrange creates a local coordinate system for its children
- Offsets in primitives are relative to their immediate parent Arrange
- Gap values control spacing between children in the arrangement direction
- Direction 'h' = horizontal layout, 'v' = vertical layout

Structural Examples:
```
Simple: (Rectangle 3 4 blue none 1 0 0)
Flat arrangement: (Arrange h rect1 rect2 2)
Nested arrangement: (Arrange h (Arrange v rect1 rect2 1) ellipse1 3)
Complex nesting: (Arrange v 
                    (Arrange h rect1 rect2 2) 
                    (Arrange h ellipse1 ellipse2 1) 
                    4)
```

Movement Strategy:
- To move a primitive: adjust its x_offset/y_offset OR restructure its parent Arrange
- To reposition groups: modify the Arrange structure or create new nested arrangements
- Gaps affect relative positioning between siblings in the same Arrange

Strategic Approach:
1. **Understand Intent**: Each VLM action represents a desired visual change. Think about WHY the VLM suggested these actions.
2. **Consider Alternatives**: Multiple expression structures might achieve the same visual effect. Choose the most elegant one.
3. **Structural Flexibility**: You may completely restructure the expression if it better achieves the visual goals.
4. **Holistic Optimization**: Consider how changes to one primitive affect the overall composition and layout.

Provide your response in this format:
<think>
1. **Current Visual Analysis**: Describe what the current expression produces visually. How are primitives positioned and arranged?

2. **VLM Intent Analysis**: For each VLM action, analyze what visual effect it's trying to achieve. Don't just think "MOVE_RIGHT means increase x_offset" - think "the primitive needs to appear more to the right relative to other elements."

3. **Strategic Options**: Consider multiple ways to achieve the desired visual effect:
   - Direct parameter modification (naive approach)
   - Rearranging Arrange structures
   - Changing groupings or nesting
   - Adjusting gaps or arrangement directions
   - Redistributing offsets across different primitives

4. **Optimal Solution**: Choose the approach that results in the cleanest, most maintainable expression while achieving the visual goal.

5. **Implementation Plan**: Explain exactly how you'll modify the expression structure.
</think>

<answer>(Modified tinySVG expression)</answer>

Key Principles:
- **Effect over Action**: Focus on achieving the visual result, not mechanically following instructions
- **Structural Intelligence**: Consider whether reorganizing Arrange operations would be more effective than parameter tweaks
- **Compositional Thinking**: Changes should enhance overall visual harmony, not just address individual primitives
- **Elegance**: Prefer simpler, cleaner expressions that achieve the same visual effect
- **Maintain Validity**: All parameters must stay within valid ranges, syntax must be correct
- **Preserve Intent**: Don't lose the core visual relationships while refactoring

Remember: The VLM actions are SUGGESTIONS for visual changes, not rigid commands. Your job is to achieve those visual changes through the best possible expression design.

INPORTANT:
- Strictly follow the <answer> ... </answer> format.
"""