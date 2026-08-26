import os
import uuid
import torch
from diffusers import StableDiffusionPipeline

from config import (
    GENERATED_DIR,
    MODEL_ID,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_SEED,
    DEFAULT_IMAGE_SIZE
)

pipe = None


def get_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_pipeline():
    global pipe

    if pipe is not None:
        return pipe

    device, dtype = get_device()
    print(f"Loading Stable Diffusion on device: {device}")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    return pipe


def generate_image(prompt, class_name="Unknown", negative_prompt=None):
    os.makedirs(GENERATED_DIR, exist_ok=True)

    pipeline = load_pipeline()
    device, _ = get_device()

    if not negative_prompt:
        negative_prompt = DEFAULT_NEGATIVE_PROMPT

    width, height = DEFAULT_IMAGE_SIZE

    generator = torch.Generator(device=device).manual_seed(DEFAULT_SEED)

    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=DEFAULT_STEPS,
        guidance_scale=DEFAULT_GUIDANCE_SCALE,
        width=width,
        height=height,
        num_images_per_prompt=1,
        generator=generator
    ).images[0]

    filename = f"{class_name}_{uuid.uuid4().hex[:8]}.png"
    save_path = os.path.join(GENERATED_DIR, filename)

    image.save(save_path)

    return {
        "filename": filename,
        "save_path": save_path,
        "relative_url": f"/static/generated/{filename}"
    }