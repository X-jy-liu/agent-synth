from PIL import Image, ImageDraw

def render_program(program, size=(100, 100)):
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)

    # Handle empty or invalid input early
    if not program or not isinstance(program, dict) or "type" not in program:
        return img  # return empty canvas

    def draw_node(node):
        if node["type"] == "Circle":
            x, y, r = node["x"], node["y"], node["r"]
            draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
        elif node["type"] == "Square":
            x, y, s = node["x"], node["y"], node["size"]
            draw.rectangle([x, y, x + s, y + s], fill=255)

    if program["type"] == "Add":
        for child in program["children"]:
            draw_node(child)
    else:
        draw_node(program)

    return img
