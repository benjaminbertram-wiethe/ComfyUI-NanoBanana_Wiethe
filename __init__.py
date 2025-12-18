# Import BananaNode from banana_node.py
from .banana_node import BananaNode
# Import TransparentImageNode from ratio_utils.py
from .ratio_utils import TransparentImageNode

# Node class name to Python class mapping
NODE_CLASS_MAPPINGS = {
    "BananaNode": BananaNode,
    "TransparentImageNode": TransparentImageNode
}

# Node display names in ComfyUI menu
NODE_DISPLAY_NAME_MAPPINGS = {
    "BananaNode": "Banana Gemini Gen",
    "TransparentImageNode": "Banana Transparent Generator"
}