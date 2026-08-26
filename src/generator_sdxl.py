import os
import uuid
import torch
from diffusers import DiffusionPipeline

from config import (
    GENERATED_DIR,
    SDXL_MODEL_ID,
    DEFAULT_NEGATIVE_PROMPT,
    SDXL_IMAGE_SIZE,
    SDXL_STEPS,
    SDXL_GUIDANCE_SCALE,
    SDXL_SEED
)

pipe = None


def get_device():
    """
    Detects the best available device.
    On your Mac, this should return MPS if available.
    """
    if torch.cuda.is_available():
        return "cuda", torch.float16

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32

    return "cpu", torch.float32


def load_sdxl_pipeline():
    """
    Loads SDXL only once and keeps it in memory.
    This prevents reloading the model every time you click Generate.
    """
    global pipe

    if pipe is not None:
        return pipe

    device, dtype = get_device()

    print(f"Loading SDXL model on device: {device}")

    pipe = DiffusionPipeline.from_pretrained(
        SDXL_MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True
    )

    pipe = pipe.to(device)

    # Memory optimizations
    pipe.enable_attention_slicing()

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    return pipe


def generate_image_sdxl(prompt, class_name="Unknown", negative_prompt=None):
    """
    Generate one image using SDXL and save it into static/generated/.
    """
    os.makedirs(GENERATED_DIR, exist_ok=True)

    pipeline = load_sdxl_pipeline()
    device, _ = get_device()

    if not negative_prompt:
        negative_prompt = DEFAULT_NEGATIVE_PROMPT

    width, height = SDXL_IMAGE_SIZE

    # For MPS, CPU generator is usually more stable
    if device == "mps":
        generator = torch.Generator(device="cpu").manual_seed(SDXL_SEED)
    else:
        generator = torch.Generator(device=device).manual_seed(SDXL_SEED)

    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=SDXL_STEPS,
        guidance_scale=SDXL_GUIDANCE_SCALE,
        num_images_per_prompt=1,
        generator=generator
    ).images[0]

    clean_class_name = class_name.replace(" ", "_").replace("/", "_")
    filename = f"{clean_class_name}_sdxl_{uuid.uuid4().hex[:8]}.png"
    save_path = os.path.join(GENERATED_DIR, filename)

    image.save(save_path)

    return {
        "filename": filename,
        "save_path": save_path,
        "relative_url": f"/static/generated/{filename}"
    }