# ImageCaption 📝

ImageCaption is a command-line utility written in Python that generates natural language descriptions for images using AI.

## Key Features
- AI-powered image caption generation
- Progress tracking with visual feedback
- User-friendly command-line interface
- Clear error handling and status messages
- Support for common image formats

## Commands
- `caption`: Starts the image captioning process
- `help`: Shows the welcome message and available commands
- `exit`: Exits the application

## Technical Details
- Uses Vision Transformer (ViT) + GPT2 model for caption generation
- Supports GPU acceleration when available
- Provides real-time progress updates during processing
- Handles errors gracefully with clear feedback
- Interactive command prompt using Rich library
- Supports PNG, JPEG, and WebP image formats

The application combines state-of-the-art AI models with an intuitive interface to generate natural language descriptions of images. The interface provides clear visual feedback throughout the process and handles potential errors with informative messages.
