# JanusScan AI - Image Authentication & Analysis

JanusScan AI is a powerful Streamlit application designed to analyze images for authenticity, detect AI-generated content, identify smartphone sources, and analyze image edits. It's particularly focused on solving the problem of images that have lost metadata through messaging platforms like WhatsApp.

## Features

- **AI Image Detection**: Identifies images created by models like DALL-E, Midjourney, Stable Diffusion, and other AI generators.
- **Device Source Detection**: Detects if an image was taken with an iPhone, Samsung Galaxy, Google Pixel, or other smartphone cameras.
- **Edit Detection & Visualization**: Identifies manipulations, edits, and shows what changes have been made to an image.
- **Metadata Recovery**: Recovers lost metadata from images that have been sent through messaging platforms.
- **AI Text Detection**: Specializes in detecting OpenAI's text generation in images, identifying synthetic text.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/JanusScan_AI.git
cd JanusScan_AI

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the Streamlit application
streamlit run app.py
```

Then open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501).

## How It Works

JanusScan AI employs a suite of state-of-the-art techniques to analyze images:

1. **AI Detection**: Uses machine learning models and statistical analysis to detect patterns common in AI-generated images.
2. **Device Detection**: Combines metadata analysis and device-specific image signatures to identify source devices.
3. **Edit Detection**: Employs advanced computer vision to detect inconsistencies in noise patterns, compression artifacts, and color profiles.
4. **Metadata Recovery**: Uses AI to reconstruct probable metadata even when original metadata is lost through messaging platforms.
5. **Text Analysis**: Detects artificial text created by OpenAI and other AI models through pattern recognition.

## Solving the WhatsApp Problem

One of JanusScan's key features is addressing the metadata loss that occurs when images are sent through messaging platforms like WhatsApp. When metadata is stripped from images, JanusScan employs:

- Noise pattern analysis
- Color profile detection
- Compression signature identification
- Device-specific artifacts recognition

These techniques allow JanusScan to recover critical information about the image source even when traditional metadata is unavailable.

## Requirements

- Python 3.8+
- Streamlit
- TensorFlow/PyTorch
- OpenCV
- Various image processing libraries (see requirements.txt)

## Limitations

- The accuracy of detection depends on image quality and resolution
- Some highly sophisticated AI-generated images may not be detected
- Device detection is most reliable for recent smartphone models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the license included in this repository.

## Acknowledgements

- Inspired by sites like [Content Credentials](https://contentcredentials.org/) and [AI or Not](https://www.aiornot.com/)
- Uses multiple open-source computer vision and machine learning libraries