import pathlib
from PIL import Image


from diffusers import StableDiffusionImg2ImgPipeline

DATAPATH = pathlib.Path(__file__).parent / "data"

# load the pipeline
device = "cpu"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
).to(device)


def image_retrieval(id_number: int) -> Image:
    """Retrieve an image from the data based based on an id number

    Args:
        id_number (int): image identification number

    Returns:
        Image:
    """
    image_num = "flower" + str(id_number) + ".jpg"
    LOCALPATH = DATAPATH / image_num
    image = Image.open(LOCALPATH).convert("RGB")
    return image


def image_pipe(
    prompt: str,
    initial_image: Image,
    strength: float = 0.5,
    guidance: float = 5,
) -> Image:
    """Image pipe for the hugging face image to image transfer
    Args:
        prompt (str): user defined prompt
        initial_image (Image): image selected by id number
        strength (float, optional): _description_. Defaults to 0.5.
        guidance (float, optional): _description_. Defaults to 5.
    Returns:
        Image: image modified by the user prompt.
    """
    modify_image = pipe(
        prompt=prompt,
        image=initial_image,
        strength=strength,
        guidance_scale=guidance,
    )
    return modify_image.images[0]
