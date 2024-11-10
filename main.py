#!/usr/bin/env python3
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt
from rich.markdown import Markdown

console = Console()

def show_welcome():
    welcome_text = """
# 📝 ImageCaption

Welcome to ImageCaption! This tool helps you generate natural language captions for your images using AI.

## Commands:
- `caption`: Generate a caption for an image
- `help`: Show this help message
- `exit`: Exit the program

## Supported Formats:
- PNG
- JPEG/JPG
- WebP
"""
    console.print(Markdown(welcome_text))

def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, feature_extractor, tokenizer, device

def generate_caption(image_path: str, model, feature_extractor, tokenizer, device) -> str:
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        generated_ids = model.generate(
            pixel_values,
            max_length=30,
            num_beams=4,
            return_dict_in_generate=True
        )

        generated_text = tokenizer.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
        return generated_text.strip()

    except Exception as e:
        console.print(f"[red]Error during caption generation: {str(e)}[/]")
        return None

def caption_command(model, feature_extractor, tokenizer, device):
    input_path = Prompt.ask("Enter image path")
    if not os.path.exists(input_path):
        console.print("[red]Error: Input file does not exist[/]")
        return

    console.print(Panel.fit(
        f"[bold green]Image Caption Generator[/]\n"
        f"Processing: [cyan]{os.path.basename(input_path)}[/]",
        border_style="green"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Generating caption...", total=100)
        progress.update(task, advance=50)
        
        caption = generate_caption(input_path, model, feature_extractor, tokenizer, device)
        progress.update(task, advance=50)

        if caption:
            console.print(Panel.fit(
                f"[bold green]Generated Caption:[/]\n[cyan]{caption}[/]",
                border_style="green"
            ))
        else:
            console.print("[red]Caption generation failed! ❌[/]")

def main():
    show_welcome()
    
    console.print("\n[yellow]Loading AI model... Please wait...[/]")
    model, feature_extractor, tokenizer, device = load_model()
    console.print("[green]Model loaded successfully! ✨[/]")
    
    while True:
        command = Prompt.ask("\nEnter command", choices=["caption", "help", "exit"])
        
        if command == "caption":
            caption_command(model, feature_extractor, tokenizer, device)
        elif command == "help":
            show_welcome()
        elif command == "exit":
            console.print("[yellow]Thanks for using ImageCaption! Goodbye! 👋[/]")
            break

if __name__ == "__main__":
    main()
