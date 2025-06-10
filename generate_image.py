#!/usr/bin/env python3
"""
brainrot_image_generation
A script for generating images from text prompts using Hugging Face Diffusers.
"""

import argparse
import os
import random
from pathlib import Path
from diffusers import StableDiffusionPipeline
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an image from a text prompt using Stable Diffusion"
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="The text prompt to generate an image for"
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


def generate_image(args):
    # Optionally set seed
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Using seed: {args.seed}")

    # Load the Stable Diffusion pipeline
    print("Loading Stable Diffusion pipeline...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)

    # Generate image
    print(f"Generating image ({args.width}x{args.height}, {args.steps} steps, scale={args.scale})...")
    image = pipe(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.scale
    ).images[0]

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Image saved to {output_path.resolve()}")


def main():
    args = parse_args()
    generate_image(args)


if __name__ == "__main__":
    main()