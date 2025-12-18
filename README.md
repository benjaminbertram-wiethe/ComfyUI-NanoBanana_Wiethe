# ComfyUI-Banana-Node

A custom node for ComfyUI that uses Google's Gemini 2.5 Flash Image and Gemini 3 Pro Image Preview APIs to generate images.

New: Online invocation
[Invincible banana2 🍌 0.2/image](https://www.runninghub.ai/post/1991513043091857410/?inviteCode=rh-v1118)
[Tutorial](https://www.bilibili.com/video/BV1ByU5BbEqS/?share_source=copy_web&vd_source=e82febcd63de4c3684cb2d737bbe5050)

- Supports resolution scaling for 1K, 2K, 4K, 8K, etc.
## Features

- Uses Google Gemini 2.5 Flash Image Preview model
- Supports multiple input images
- Customizable prompts
- Multiple output aspect ratios
- Complete error handling and logging
- New: Image Ratio Adjuster node — supports crop, pad, stretch
- New: Resolution Scaler node — supports 1K, 2K, 4K, 8K resolution scaling

## Installation

1. Clone this repository into your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI-Banana-Node.git
```

2. Install dependencies:
```bash
cd ComfyUI-Banana-Node
pip install google-generativeai pillow numpy torch
```

3. Get your API key:
- Visit Google AI Studio: https://aistudio.google.com/app/apikey
- Bind a Visa or MasterCard to receive $300 in free credits

4. Restart ComfyUI

## Usage

### Banana Gemini Gen Node
1. In ComfyUI, find the "Banana" category and add the "Banana Gemini Gen" node
2. Connect input image(s)
3. Set the prompt (describe the image you want to generate)
4. Choose the output aspect ratio
5. Run the workflow

### Banana Ratio Adjuster Node
1. In ComfyUI, find the "Banana Node/Ratio" category and add the "Banana Ratio Adjuster" node
2. Connect input image(s)
3. Choose target ratio (1:1, 4:3, 16:9 or custom)
4. Choose adjustment method:
   - crop: crop the image to match the target ratio
   - pad: pad the image to match the target ratio
   - stretch: stretch the image to match the target ratio
5. If pad mode is selected, set the padding color
6. Run the workflow

### Banana Resolution Scaler Node
1. In ComfyUI, find the "Banana Node/Ratio" category and add the "Banana Resolution Scaler" node
2. Connect input image(s)
3. Choose target resolution (1K, 2K, 4K, 8K or custom)
4. Choose whether to maintain aspect ratio
5. Run the workflow

## Input Parameters

### Banana Gemini Gen Node
- model: model selection (currently supports gemini-2.5-flash-image & gemini-3-pro-image-preview)
- prompt: text prompt describing the image you want to generate
- size: output image aspect ratio
- input_image: input image (required)

### Banana Ratio Adjuster Node
- image: input image
- target_ratio: target ratio (1:1, 4:3, 16:9 or custom)
- resize_method: adjustment method (crop, pad, stretch)
- custom_width: custom width ratio (when using custom ratio)
- custom_height: custom height ratio (when using custom ratio)
- pad_color_r/g/b: RGB values of the padding color (when using pad)

### Banana Resolution Scaler Node
- image: input image
- target_resolution: target resolution (1K, 2K, 4K, 8K, custom)
- maintain_ratio: whether to maintain aspect ratio
- custom_size: custom size (when using custom resolution)

## Output

### Banana Gemini Gen Node
- image: generated image
- revised_prompt: revised prompt (currently N/A)
- image_url: image URL (currently N/A)

### Banana Ratio Adjuster Node
- image: adjusted image
- width: output image width
- height: output image height
- ratio: final aspect ratio

### Banana Resolution Scaler Node
- image: scaled image
- width: output image width
- height: output image height

## Notes

- A valid Google AI API key is required
- Ensure your network connection is stable
- API calls may incur costs; please check Google AI pricing

## Troubleshooting

1. API key error:
   - Verify that the API key is correct
   - Ensure the API key has sufficient permissions and quota

3. Dependency errors:
   - Ensure all required Python packages are installed
   - Try reinstalling dependencies

## License

MIT License

## Contributing

Issues and pull requests are welcome!

## Changelog

### v1.1.0
- Added Image Ratio Adjuster node (Banana Ratio Adjuster)
- Added Resolution Scaler node (Banana Resolution Scaler)
- Supports multiple ratio adjustment methods (crop, pad, stretch)
- Supports 1K, 2K, 4K, 8K resolution scaling
- Enhanced ratio calculation and image processing utilities

### v1.0.0
- Initial release
- Supports Gemini 2.5 Flash Image Preview API
- Basic image generation features