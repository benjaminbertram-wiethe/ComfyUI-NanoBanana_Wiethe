import torch
import numpy as np
from PIL import Image, ImageDraw
import math

def tensor2pil(image):
    """Convert tensor to list of PIL images"""
    if len(image.shape) == 4:
        # Batch dimension exists
        return [Image.fromarray(np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)) for img in image]
    else:
        # Single image
        return [Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))]

def pil2tensor(images):
    """Convert list of PIL images to tensor"""
    if isinstance(images, list):
        tensors = [torch.from_numpy(np.array(img).astype(np.float32) / 255.0) for img in images]
        return torch.stack(tensors, dim=0)
    else:
        return torch.from_numpy(np.array(images).astype(np.float32) / 255.0).unsqueeze(0)

class TransparentImageNode:
    """Center input image on a proportional background"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": (["1K", "2K", "4K"], {"default": "2K"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    FUNCTION = "generate_centered_image"
    CATEGORY = "Banana Node"
    
    def generate_centered_image(self, image, resolution):
        pil_images = tensor2pil(image)
        result_images = []
        
        resolution_map = {"1K": 1024, "2K": 2048, "4K": 4096}
        target_size = resolution_map[resolution]
        
        for pil_img in pil_images:
            orig_width, orig_height = pil_img.size
            aspect_ratio = orig_width / max(1, orig_height)
            
            # 1. Calculate target canvas size based on original aspect ratio and target resolution
            if aspect_ratio > 1:
                proportional_canvas_width = int(round(target_size))
                proportional_canvas_height = max(1, int(round(target_size / aspect_ratio)))
            else:
                proportional_canvas_height = int(round(target_size))
                proportional_canvas_width = max(1, int(round(target_size * aspect_ratio)))
            
            # 2. Final canvas size must accommodate original image, so take max of calculated and original
            canvas_width = max(proportional_canvas_width, orig_width)
            canvas_height = max(proportional_canvas_height, orig_height)
            
            # 3. Create black background canvas
            canvas = Image.new('RGB', (canvas_width, canvas_height), 'black')
            
            # 4. Calculate offset to center original image
            x_offset = (canvas_width - orig_width) // 2
            y_offset = (canvas_height - orig_height) // 2
            
            # 5. Paste original image directly without scaling
            canvas.paste(pil_img, (x_offset, y_offset))

            # --- Draw arrows at four corners ---
            # Only draw if there are borders
            if x_offset > 0 or y_offset > 0:
                draw = ImageDraw.Draw(canvas)
                
                # Dynamically adjust arrow thickness
                arrow_color = "black"
                # Based on smaller canvas dimension (smaller divisor = thicker, larger min width)
                arrow_width = max(4, int(min(canvas_width, canvas_height) / 256))

                # Define four corners of original image
                img_tl = (x_offset, y_offset)
                img_tr = (x_offset + orig_width, y_offset)
                img_bl = (x_offset, y_offset + orig_height)
                img_br = (x_offset + orig_width, y_offset + orig_height)
                
                # Define four corners of canvas (use -1 to ensure within canvas)
                canvas_tl = (0, 0)
                canvas_tr = (canvas_width - 1, 0)
                canvas_bl = (0, canvas_height - 1)
                canvas_br = (canvas_width - 1, canvas_height - 1)
                
                # Draw four arrows
                _draw_arrow(draw, img_tl, canvas_tl, fill=arrow_color, width=arrow_width)
                _draw_arrow(draw, img_tr, canvas_tr, fill=arrow_color, width=arrow_width)
                _draw_arrow(draw, img_bl, canvas_bl, fill=arrow_color, width=arrow_width)
                _draw_arrow(draw, img_br, canvas_br, fill=arrow_color, width=arrow_width)
            
            result_images.append(canvas)
        
        return (pil2tensor(result_images),)


def _draw_arrow(draw, start, end, fill, width):
    """Draw an arrow line from start point to end point on the given canvas (draw)."""
    
    # Shorten arrow shaft to 80% to ensure arrowhead is always fully visible
    shaft_end_x = start[0] + 0.8 * (end[0] - start[0])
    shaft_end_y = start[1] + 0.8 * (end[1] - start[1])
    shaft_end = (shaft_end_x, shaft_end_y)
    
    # Draw shortened main line segment
    draw.line([start, shaft_end], fill=fill, width=width)

    # Arrowhead line length proportional to main line width
    # (larger multiplier = longer, larger min length)
    arrowhead_len = max(30, width * 8)
    arrowhead_angle = math.pi / 6

    # Arrow pointing angle remains unchanged (always points to original 'end')
    angle = math.atan2(end[1] - start[1], end[0] - start[0])

    # Calculate endpoint coordinates for arrowhead lines (relative to shaft_end)
    p1_x = shaft_end[0] - arrowhead_len * math.cos(angle - arrowhead_angle)
    p1_y = shaft_end[1] - arrowhead_len * math.sin(angle - arrowhead_angle)

    p2_x = shaft_end[0] - arrowhead_len * math.cos(angle + arrowhead_angle)
    p2_y = shaft_end[1] - arrowhead_len * math.sin(angle + arrowhead_angle)

    # Draw arrowhead lines from shaft_end
    draw.line([shaft_end, (p1_x, p1_y)], fill=fill, width=width)
    draw.line([shaft_end, (p2_x, p2_y)], fill=fill, width=width)