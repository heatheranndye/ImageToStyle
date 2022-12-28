from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ml_imagetoimage import DATAPATH, image_retrieval, image_pipe

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


class User_input(BaseModel):
    id_num: str
    prompt: str


@app.post("/items/")
def create_item(values: User_input):
    return values


@app.post("/generate")
def generate_image(values: User_input):
    initial_image = image_retrieval(values.id_num)
    image = image_pipe(values.prompt, initial_image)
    image.save("image.png")
    return FileResponse("image.png")
