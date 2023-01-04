from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from photo_mod.ml_imagetoimage import image_retrieval, image_pipe

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


class User_input(BaseModel):
    id_number: str
    prompt: str


@app.post("/items/")
def create_item(values: User_input):
    return values


@app.post("/generate")
def generate_image(values: User_input):
    """Generate a transformed image from the selected base image
    identified by ID number and the given prompt

    Args:
        values (User_input): id_num for the image and
        a user defined prompt.

    Returns:
        _type_: _description_
    """
    initial_image = image_retrieval(values.id_number)
    image = image_pipe(values.prompt, initial_image)
    image.save("image.png")
    return FileResponse("image.png")
