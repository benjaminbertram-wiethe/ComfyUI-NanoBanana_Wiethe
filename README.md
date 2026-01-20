# ComfyUI-Banana-Node

A custom node for ComfyUI that uses Google's Vertex AI Gemini models to generate images.

## Features

- Uses Google Vertex AI Gemini models (gemini-2.5-flash-preview-image-generation, gemini-2.0-flash-exp)
- **Vertex AI authentication** with service account JSON
- Supports multiple input images (up to 14)
- Customizable prompts
- Multiple output aspect ratios
- Resolution options: 1K, 2K, 4K
- Complete error handling and logging
- Image Ratio Adjuster node — supports crop, pad, stretch
- Resolution Scaler node — supports 1K, 2K, 4K, 8K resolution scaling

## Installation

1. Clone this repository into your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI-Banana-Node.git
```

2. Install dependencies:
```bash
cd ComfyUI-Banana-Node
pip install google-auth google-auth-httplib2 pillow numpy torch requests
```

3. Set up Vertex AI credentials:
   - Create a Google Cloud project with Vertex AI API enabled
   - Create a Service Account with the **"Vertex AI User"** role (`roles/aiplatform.user`)
   - Download the Service Account JSON key file
   - Note your Project ID and preferred region

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
- **model**: model selection (gemini-2.5-flash-preview-image-generation, gemini-2.0-flash-exp)
- **prompt**: text prompt describing the image you want to generate
- **aspect_ratio**: output image aspect ratio (Automatic, 1:1, 2:3, 3:2, etc.)
- **resolution**: output resolution (1K, 2K, 4K)
- **seed**: random seed for reproducibility
- **service_account_json**: path to your Google Cloud service account JSON file
- **project_id**: your Google Cloud Project ID
- **region**: Vertex AI region (us-central1, europe-west1, etc.)
- **image1-14**: input images (optional, up to 14 images)

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

- A valid Google Cloud service account with Vertex AI User role is required
- Ensure your network connection is stable
- API calls may incur costs; please check Google Cloud Vertex AI pricing

## Troubleshooting

1. Authentication error:
   - Verify that the service account JSON file path is correct
   - Ensure the service account has the "Vertex AI User" role
   - Check that the project ID is correct

2. Region error:
   - Make sure the selected region has Vertex AI and the Gemini model available

3. Dependency errors:
   - Ensure all required Python packages are installed: `pip install google-auth google-auth-httplib2 requests`

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