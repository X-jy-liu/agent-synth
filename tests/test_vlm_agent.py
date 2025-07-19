import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
from vlm_agent import call_vlm_agent

class TestVLMWithMock(unittest.TestCase):
    def setUp(self):
        # Create two dummy grayscale images
        self.current_img = Image.new("L", (100, 100), color=0)
        self.target_img = Image.new("L", (100, 100), color=255)
        self.program_code = "(Add (Circle x=30 y=30 r=10))"

    @patch("openai.chat.completions.create")
    def test_call_vlm_agent_returns_valid_edit(self, mock_openai_call):
        # Simulated JSON response from GPT-4V
        fake_edit = {
            "action": "insert_child",
            "target_path": [],
            "new_node": {
                "type": "Square",
                "x": 50,
                "y": 50,
                "size": 20
            }
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = str(fake_edit).replace("'", '"')  # make valid JSON string
        mock_openai_call.return_value = mock_response

        result = call_vlm_agent(self.current_img, self.target_img, self.program_code)

        self.assertIn("thought", result)
        self.assertIn("edit", result)
        self.assertEqual(result["edit"]["action"], "insert_child")
        self.assertEqual(result["edit"]["new_node"]["type"], "Square")

if __name__ == "__main__":
    unittest.main()
