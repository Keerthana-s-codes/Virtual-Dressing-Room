# backend/ootd_infer.py
import threading
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from PIL import Image
import numpy as np

_lock = threading.Lock()
_loaded = False
_pipe = None

# CPU vs CUDA
DEVICE = "cpu"
DTYPE = torch.float32

# model ids (public)
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET = "xinsir/controlnet-openpose-sdxl-1.0"

def _load_models():
    global _loaded, _pipe
    with _lock:
        if _loaded:
            return
        print("OOT: Loading ControlNet and SDXL (this will download ~ several GB).")
        controlnet = ControlNetModel.from_pretrained(CONTROLNET, torch_dtype=DTYPE)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            BASE_MODEL,
            controlnet=controlnet,
            torch_dtype=DTYPE,
            safety_checker=None,   # optional: disable safety checker if you prefer
        )
        pipe.to(DEVICE)
        # memory-saver for CPU
        pipe.enable_attention_slicing()
        _pipe = pipe
        _loaded = True
        print("OOT: models loaded.")

def run_ootd(person_img: Image.Image, cloth_img: Image.Image, prompt="A realistic person wearing the provided garment, studio photo", steps: int = 30):
    """
    person_img, cloth_img: PIL.Image (RGB)
    returns: PIL.Image
    """
    global _loaded, _pipe
    if not _loaded:
        _load_models()  # lazy load, will block while downloading/loading

    # Resize cloth/control image to what ControlNet expects
    # Many SDXL controlnet implementations expect a control image of 512x512 or same size as person
    person = person_img.convert("RGB")
    cloth = cloth_img.convert("RGB").resize((512, 512))

    # run pipeline
    out = _pipe(
        prompt=prompt,
        image=person,
        control_image=cloth,
        num_inference_steps=steps,
        guidance_scale=7.5,
    )
    return out.images[0]
