#!/usr/bin/env python3
"""
brainrot_image_generation
A script for generating images from text prompts using Hugging Face Diffusers, with logging.
"""

import argparse
import os
import random
import logging
from pathlib import Path
from diffusers import StableDiffusionPipeline
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an image from a text prompt using Stable Diffusion"
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="The text prompt to generate an image for"
    )
    parser.add_argument(
        "--template", type=str, default=None,
        help="Name of the prompt template to use"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--width", type=int, default=512,
        help="Width of the generated image"
    )
    parser.add_argument(
        "--height", type=int, default=512,
        help="Height of the generated image"
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--scale", type=float, default=7.5,
        help="Guidance scale (higher encourages following the prompt more)"
    )
    parser.add_argument(
        "--output", type=str, default="output.png",
        help="Path to save the generated image"
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug(f"Random seed set to {seed}")


def load_template(name: str):
    """
    Load a prompt template by name from the templates folder.
    """
    templates_dir = Path(__file__).parent / "templates"
    template_path = templates_dir / f"{name}.md"
    if not template_path.exists():
        logger.error(f"Template '{name}' not found in {templates_dir}")
        raise FileNotFoundError(f"Template '{name}' not found")
    content = template_path.read_text().strip()
    logger.info(f"Loaded template '{name}'")
    return content


def generate_image(args):
    # Prepare prompt
    if args.template:
        template_text = load_template(args.template)
        prompt = template_text.replace("{{PROMPT}}", args.prompt)
        logger.info(f"Using template '{args.template}' with prompt injection")
        logger.debug(f"Template content: {template_text}")
    else:
        prompt = args.prompt
        logger.info("Using raw prompt without template")

    # Optionally set seed
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Seed applied: {args.seed}")

    # Load the Stable Diffusion pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Stable Diffusion pipeline on {device}")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)

    # Generate image
    logger.info(
        f"Generating image [prompt length={len(prompt)} chars, size={args.width}x{args.height}, "
        f"steps={args.steps}, scale={args.scale}]")
    output = pipe(
        prompt=prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.scale
    )
    image = output.images[0]

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    logger.info(f"Image saved to {output_path.resolve()}")


def main():
    args = parse_args()
    try:
        generate_image(args)
    except Exception as e:
        logger.exception("An error occurred during image generation")
        raise


if __name__ == "__main__":
    main()
