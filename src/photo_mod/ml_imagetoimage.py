import torch
import requests
import pathlib
from PIL import Image
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

DATAPATH = pathlib.Path(__file__).parent / "data"

# load the pipeline
device = "cpu"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
).to(device)


def hg_image_download() -> Image:
    """original image code"""
    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image.thumbnail((768, 768))
    return init_image


def image_retrieval(id_num: int) -> Image:
    image_num = "flower" + str(id_num) + ".jpg"
    LOCALPATH = DATAPATH / image_num
    image = Image.open(LOCALPATH).convert("RGB")
    return image


prompt = "A fantasy landscape, trending on artstation"
prompt2 = "In the style of Monet"


def image_pipe(
    prompt: str,
    initial_image: Image,
    strength: float = 0.5,
    guidance: float = 5,
) -> Image:
    modify_image = pipe(
        prompt=prompt,
        image=initial_image,
        strength=strength,
        guidance_scale=guidance,
    )
    return modify_image.images[0]


# images[0].save("fantasy_landscape.png")
