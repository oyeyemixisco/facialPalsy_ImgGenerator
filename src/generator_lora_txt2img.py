import os
import uuid
import torch
from diffusers import StableDiffusionPipeline

from config import (
    MODEL_ID,
    GENERATED_DIR,
    DEFAULT_NEGATIVE_PROMPT,
    LORA_DIR,
    LORA_WEIGHT_NAME,
    LORA_SCALE
)

pipe = None


def get_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32

    return "cpu", torch.float32


def load_lora_txt2img_pipeline():
    global pipe

    if pipe is not None:
        return pipe

    device, dtype = get_device()

    print(f"Loading text-to-image LoRA pipeline on device: {device}")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None
    )

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    lora_path = os.path.join(LORA_DIR, LORA_WEIGHT_NAME)

    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")

    print(f"Loading LoRA weights: {lora_path}")

    pipe.load_lora_weights(
        LORA_DIR,
        weight_name=LORA_WEIGHT_NAME
    )

    try:
        pipe.set_adapters(["default"], adapter_weights=[LORA_SCALE])
        print(f"LoRA scale set to: {LORA_SCALE}")
    except Exception as e:
        print(f"LoRA loaded, but scale could not be set: {e}")

    return pipe


def generate_lora_text2img(
    prompt,
    class_name="Unknown",
    negative_prompt=None,
    steps=40,
    guidance_scale=8.5,
    seed=42
):
    os.makedirs(GENERATED_DIR, exist_ok=True)

    pipeline = load_lora_txt2img_pipeline()
    device, _ = get_device()

    if negative_prompt is None:
        negative_prompt = DEFAULT_NEGATIVE_PROMPT

    if device == "mps":
        generator = torch.Generator(device="cpu").manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)

    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=512,
        height=512,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator
    )

    image = result.images[0]

    clean_class = class_name.replace(" ", "_").replace("/", "_")
    filename = f"{clean_class}_lora_txt2img_{uuid.uuid4().hex[:8]}.png"
    save_path = os.path.join(GENERATED_DIR, filename)

    image.save(save_path)

    return {
        "filename": filename,
        "save_path": save_path,
        "relative_url": f"/static/generated/{filename}"
    }