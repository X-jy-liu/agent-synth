import unittest
from render_engine import render_program
from PIL import ImageChops

class TestRenderEngine(unittest.TestCase):

    def test_render_circle(self):
        program = {
            "type": "Circle",
            "x": 50,
            "y": 50,
            "r": 10
        }
        img = render_program(program)
        self.assertEqual(img.size, (100, 100))
        self.assertGreater(sum(img.getdata()), 0, "Circle should produce non-zero pixels")

    def test_render_square(self):
        program = {
            "type": "Square",
            "x": 30,
            "y": 30,
            "size": 20
        }
        img = render_program(program)
        self.assertEqual(img.size, (100, 100))
        self.assertGreater(sum(img.getdata()), 0, "Square should produce non-zero pixels")

    def test_render_add(self):
        program = {
            "type": "Add",
            "children": [
                {"type": "Circle", "x": 30, "y": 30, "r": 10},
                {"type": "Square", "x": 50, "y": 50, "size": 20}
            ]
        }
        img = render_program(program)
        self.assertEqual(img.size, (100, 100))
        self.assertGreater(sum(img.getdata()), 0, "Combined shape should produce non-zero pixels")

    def test_render_empty_program(self):
        with self.assertRaises(KeyError):
            render_program({})

if __name__ == "__main__":
    unittest.main()
