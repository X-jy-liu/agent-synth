from render_engine import render_program
import matplotlib.pyplot as plt

program = {
    "type": "Add",
    "children": [
        {"type": "Circle", "x": 30, "y": 30, "r": 10},
        {"type": "Square", "x": 50, "y": 50, "size": 20}
    ]
}

img = render_program(program)

plt.imshow(img, cmap="gray")
plt.title("Rendered Program")
plt.axis("off")
plt.show()
