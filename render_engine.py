from PIL import Image, ImageDraw

def render_program(program, size=(100, 100)):
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)

    def draw_node(node):
        if not isinstance(node, dict):
            return
        node_type = node.get("type")

        if node_type == "Circle":
            x, y, r = node["x"], node["y"], node["r"]
            draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

        elif node_type == "Square":
            x, y, s = node["x"], node["y"], node["s"]
            draw.rectangle([x, y, x + s, y + s], fill=255)

        # Recurse into children if present
        if "children" in node:
            for child in node["children"]:
                draw_node(child)

    draw_node(program)
    return img