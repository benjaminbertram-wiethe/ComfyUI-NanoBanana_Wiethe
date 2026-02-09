import os
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import logging
import base64
import time
import requests
import re
import random

# --- Logging Configuration ---
# We place logger acquisition in the class methods to ensure correct scope
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - BananaNode - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


# --- Helper functions for Image Conversion ---
def tensor2pil(image: torch.Tensor) -> list[Image.Image]:
    """Converts a torch tensor to a list of PIL Images."""
    batch_count = image.shape[0]
    images = []
    for i in range(batch_count):
        img_tensor = image[i]
        img_np = (img_tensor.cpu().numpy().squeeze() * 255).astype(np.uint8)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3: # HWC
            images.append(Image.fromarray(img_np, 'RGB'))
        elif len(img_np.shape) == 2: # HW (grayscale)
            images.append(Image.fromarray(img_np, 'L'))
        else: # Fallback for other formats
            images.append(Image.fromarray(img_np))
    return images

def pil2tensor(images: list[Image.Image]) -> torch.Tensor:
    """Converts a list of PIL Images to a torch tensor."""
    tensors = []
    for img in images:
        img_np = np.array(img).astype(np.float32) / 255.0
        if len(img_np.shape) == 2: # Grayscale to RGB
            img_np = np.stack([img_np]*3, axis=-1)
        tensors.append(torch.from_numpy(img_np))
    return torch.stack(tensors)


class BananaNodeAPI:
    """
    ComfyUI node for image generation using Google Gemini API.
    Requires API key input directly in the node.
    """

    def add_random_variation(self, prompt, seed=0):
        """
        Add a hidden random identifier to the end of the prompt
        to ensure different results each run
        """
        if seed == 0:
            random_id = random.randint(10000, 99999)
        else:
            rng = random.Random(seed)
            random_id = rng.randint(10000, 99999)

        return f"{prompt} [variation-{random_id}]"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["gemini-2.5-flash-image","gemini-3-pro-image-preview"],{"default": "gemini-3-pro-image-preview"}),
                "prompt": ("STRING", {"multiline": True, "default": "Combine the features of all input images into a single new image."}),
                "aspect_ratio": (["Automatic", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"], {"default": "Automatic"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Random seed, changing this value forces content regeneration"
                }),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "base_url": ("STRING", {"multiline": False, "default": ""}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6": ("IMAGE",),
                "image7": ("IMAGE",),
                "image8": ("IMAGE",),
                "image9": ("IMAGE",),
                "image10": ("IMAGE",),
                "image11": ("IMAGE",),
                "image12": ("IMAGE",),
                "image13": ("IMAGE",),
                "image14": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "revised_prompt", "image_url")
    FUNCTION = "generate"
    CATEGORY = "Banana"

    def generate(self, model: str, prompt: str, aspect_ratio: str, resolution: str, seed: int = 0, api_key: str = "", base_url: str = "",
                 image1: torch.Tensor = None, image2: torch.Tensor = None, image3: torch.Tensor = None, image4: torch.Tensor = None,
                 image5: torch.Tensor = None, image6: torch.Tensor = None, image7: torch.Tensor = None, image8: torch.Tensor = None,
                 image9: torch.Tensor = None, image10: torch.Tensor = None, image11: torch.Tensor = None, image12: torch.Tensor = None,
                 image13: torch.Tensor = None, image14: torch.Tensor = None):
        # --- Initialize logger at function start ---
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - BananaNode - %(levelname)s - %(message)s', force=True)
        logger = logging.getLogger(__name__)

        # Add random variation factor to prompt
        varied_prompt = self.add_random_variation(prompt, seed)
        logger.info(f"Random seed: {seed}")

        # Validate API key
        api_key = api_key.strip() if api_key else ""
        if not api_key:
            raise ValueError("Google AI API Key is required. Please enter your API key in the node.")

        # Set timeout based on resolution
        timeout = 120  # Default 1K/2K: 2 minutes
        if resolution == "4K":
            timeout = 360  # 4K: 6 minutes
        logger.info(f"Timeout set to: {timeout} seconds (resolution: {resolution})")

        # Aggregate all image inputs (multiple ports + batches), order: image1 → image14
        aggregated_pil_images = []
        try:
            images_list = [image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, image11, image12, image13, image14]
            for img_tensor in images_list:
                if img_tensor is not None:
                    aggregated_pil_images.extend(tensor2pil(img_tensor))
        except Exception as e:
            raise ValueError(f"Error: Failed to parse image input - {e}")

        # if not aggregated_pil_images:
        #     raise ValueError("Error: At least one image is required (image1~image4).")

        try:
            # Handle custom endpoint from node input
            endpoint = (base_url or '').strip()
            if endpoint.endswith('/'):
                endpoint = endpoint[:-1]

            # Consistent with Gemini_Pro_Node: expose endpoint via environment variable for potential subprocesses
            if endpoint:
                os.environ['GENAI_API_ENDPOINT'] = endpoint
            else:
                if 'GENAI_API_ENDPOINT' in os.environ:
                    del os.environ['GENAI_API_ENDPOINT']

            # ========== Using REST API calls instead of SDK ==========
            # Assemble base address (use official default if empty)
            base = endpoint if endpoint else "https://generativelanguage.googleapis.com"
            if base.endswith('/'):
                base = base[:-1]

            # Path requires models/<model>:generateContent
            model_path = model if isinstance(model, str) and model.startswith('models/') else f"models/{model}"
            url = f"{base}/v1beta/{model_path}:generateContent"

            # Log input statistics
            batch_size = len(aggregated_pil_images)
            logger.info(f"Detected {batch_size} image inputs, sending as single task (REST API call).")

            # Build parts: first add image identifier text then image (PNG base64)
            parts = []
            # First add image identifiers + images
            for idx, img in enumerate(aggregated_pil_images, start=1):
                try:
                    # Add image identifier text for model reference (e.g., "Image 1", "Image 2")
                    parts.append({"text": f"[This is Image {idx}]"})
                    buf = BytesIO()
                    img.save(buf, format='PNG')
                    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    parts.append({
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": b64
                        }
                    })
                except Exception as _e:
                    logger.warning(f"Failed to encode input image (skipped): {_e}")
            # Then add user text instructions
            if varied_prompt:
                parts.append({"text": varied_prompt})

            # Build generationConfig
            generation_config = {
                "temperature": 0.4,
                "maxOutputTokens": 8192,
                "candidateCount": 1
            }

            # Build imageConfig
            image_config = {}
            if aspect_ratio != "Automatic":
                image_config["aspectRatio"] = aspect_ratio

            # Only gemini-3-pro-image-preview supports imageSize (resolution)
            image_config["imageSize"] = resolution

            if image_config:
                generation_config["imageConfig"] = image_config

            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": parts
                    }
                ],
                "generationConfig": generation_config
            }

            # Use query parameters to pass API Key (official REST supports key=...)
            params = {"key": api_key}

            # Requests proxy (uses environment variables if set)
            req_proxies = None

            response_data = None
            last_error = "Unknown error"
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    logger.info(f"[Banana|REST] Attempting API call {attempt + 1}/{max_retries}...")
                    start_time = time.time()
                    r = requests.post(url, json=payload, params=params, timeout=timeout, proxies=req_proxies, headers={"Content-Type": "application/json"})
                    end_time = time.time()
                    if r.status_code == 200:
                        response_data = r.json()
                        logger.info(f"[Banana|REST] API call successful, took: {end_time - start_time:.2f} seconds")
                        break
                    else:
                        last_error = f"HTTP {r.status_code}: {r.text[:300]}"
                        logger.warning(f"[Banana|REST] Received non-200 status code: {last_error}")
                        if attempt < max_retries - 1:
                            time.sleep(1)
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"[Banana|REST] Attempt {attempt + 1} failed: {last_error}")
                    if attempt < max_retries - 1:
                        logger.info("[Banana|REST] Network connection timeout or error, retrying in 1 second...")
                        time.sleep(1)

            if response_data is None:
                raise ValueError(f"Error: API returned no content. Last error: {last_error}")

            # Default fallback: if no image generated, pass through first input image to ensure correct port type
            fallback_tensor = pil2tensor([aggregated_pil_images[0]]) if aggregated_pil_images else torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            output_tensor = fallback_tensor
            image_url_output = "N/A"

            # Extract all returned images (including inlineData and data:image/...;base64,... in text)
            generated_pils = []
            first_image_url = None
            try:
                candidates = []
                if isinstance(response_data, dict):
                    candidates = response_data.get('candidates') or []
                for cand in candidates:
                    content = cand.get('content') if isinstance(cand, dict) else None
                    parts_list = []
                    if isinstance(content, dict):
                        parts_list = content.get('parts') or []
                    elif isinstance(content, list):
                        parts_list = content
                    for part in parts_list or []:
                        if not isinstance(part, dict):
                            continue
                        # 1) inlineData images
                        inline_data = part.get('inlineData') or part.get('inline_data')
                        if isinstance(inline_data, dict):
                            data = inline_data.get('data')
                            mime = inline_data.get('mimeType') or inline_data.get('mime_type') or 'image/png'
                            if data:
                                try:
                                    img_bytes = data if isinstance(data, bytes) else base64.b64decode(data)
                                    pil = Image.open(BytesIO(img_bytes)).convert('RGB')
                                    generated_pils.append(pil)
                                    if first_image_url is None:
                                        # Convert to data URL for third output port
                                        b64_out = base64.b64encode(img_bytes).decode('utf-8')
                                        first_image_url = f"data:{mime};base64,{b64_out}"
                                except Exception:
                                    pass
                        # 2) data:image/...;base64,... in text
                        txt = part.get('text')
                        if isinstance(txt, str) and ('data:image/' in txt):
                            try:
                                # Support multiple data URLs
                                matches = re.findall(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', txt)
                                for url in matches:
                                    try:
                                        _, b64data = url.split(',', 1)
                                        img_bytes = base64.b64decode(b64data)
                                        pil = Image.open(BytesIO(img_bytes)).convert('RGB')
                                        generated_pils.append(pil)
                                        if first_image_url is None:
                                            first_image_url = url
                                    except Exception:
                                        continue
                            except Exception:
                                pass
            except Exception as e:
                logger.warning(f"Failed to extract images: {str(e)}")

            if generated_pils:
                logger.info(f"API request successful, extracted {len(generated_pils)} returned images.")
                output_tensor = pil2tensor(generated_pils)
                image_url_output = first_image_url or "N/A"
            else:
                logger.warning("No generated image data found in API response, passing through input image as output.")

            # Extract model returned text as revised_prompt (if exists)
            try:
                texts = []
                if isinstance(response_data, dict):
                    for cand in (response_data.get('candidates') or []):
                        content = cand.get('content') if isinstance(cand, dict) else None
                        parts_list = content.get('parts') if isinstance(content, dict) else []
                        for p in parts_list or []:
                            if isinstance(p, dict):
                                t = p.get('text')
                                if t:
                                    texts.append(t)
                revised_prompt_output = (" ".join(texts)).strip() if texts else "N/A"
            except Exception:
                revised_prompt_output = "N/A"

            return (output_tensor, revised_prompt_output, image_url_output)

        except Exception as e:
            # Ensure logger is available in exception handling
            logger.error(f"Error occurred: {e}")
            raise
