# Import BananaNode from banana_node.py
from .banana_node_api import BananaNodeAPI
from .banana_node_vertex import BananaNodeVertexAI
# Import TransparentImageNode from ratio_utils.py
from .ratio_utils import TransparentImageNode

# Node class name to Python class mapping
NODE_CLASS_MAPPINGS = {
    "BananaNodeAPI": BananaNodeAPI,
    "BananaNodeVertexAI": BananaNodeVertexAI,
    "TransparentImageNode": TransparentImageNode
}

# Node display names in ComfyUI menu
NODE_DISPLAY_NAME_MAPPINGS = {
    "BananaNodeAPI": "Nano Banana Gemini Gen via API",
    "BananaNodeVertexAI": "Nano Banana Gemini Gen via VertexAI",
    "TransparentImageNode": "Banana Transparent Generator"
}