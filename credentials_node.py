import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class VertexAICredentialsNode:
    """
    A node for configuring Vertex AI credentials and settings.
    This node outputs a credentials object that can be passed to the NanoBananaNode.
    
    Allows you to:
    - Override environment variables per-workflow
    - Configure multiple credential sets
    - Keep credentials separate from the main node
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input specification for this node.
        All fields have defaults from environment variables.
        """
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": os.getenv("VERTEX_AI_API_KEY", ""),
                    "placeholder": "Enter your Vertex AI API key"
                }),
                "use_simple_endpoint": ("BOOLEAN", {
                    "default": os.getenv("VERTEX_AI_USE_SIMPLE_ENDPOINT", "false").lower() == "true"
                }),
            },
            "optional": {
                # For simple endpoint (proxy)
                "endpoint": ("STRING", {
                    "multiline": False,
                    "default": os.getenv("VERTEX_AI_ENDPOINT", ""),
                    "placeholder": "https://your-nano-banana-endpoint.com"
                }),
                # For standard Vertex AI endpoint
                "project": ("STRING", {
                    "multiline": False,
                    "default": os.getenv("VERTEX_AI_PROJECT", ""),
                    "placeholder": "your-gcp-project-id"
                }),
                "location": ("STRING", {
                    "multiline": False,
                    "default": os.getenv("VERTEX_AI_LOCATION", "us-central1"),
                    "placeholder": "us-central1"
                }),
            }
        }

    RETURN_TYPES = ("VERTEX_CREDENTIALS",)
    RETURN_NAMES = ("credentials",)

    FUNCTION = "create_credentials"
    CATEGORY = "LLM"

    def create_credentials(self, api_key, use_simple_endpoint, endpoint="", project="", location=""):
        """
        Creates a credentials dictionary that can be passed to the NanoBananaNode.
        
        Returns:
            A dictionary containing all necessary credentials and configuration.
        """
        # Validate required fields
        if not api_key or api_key.strip() == "":
            raise ValueError("API key is required. Please enter your Vertex AI API key.")
        
        if use_simple_endpoint:
            if not endpoint or endpoint.strip() == "":
                raise ValueError("Endpoint URL is required when using simple endpoint mode.")
        else:
            if not project or project.strip() == "":
                raise ValueError("Project ID is required when using standard Vertex AI endpoint.")
            if not location or location.strip() == "":
                raise ValueError("Location is required when using standard Vertex AI endpoint.")
        
        credentials = {
            "api_key": api_key.strip(),
            "use_simple_endpoint": use_simple_endpoint,
            "endpoint": endpoint.strip() if endpoint else "",
            "project": project.strip() if project else "",
            "location": location.strip() if location else "",
        }
        
        return (credentials,)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "VertexAICredentialsNode": VertexAICredentialsNode
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "VertexAICredentialsNode": "Vertex AI Credentials"
}

