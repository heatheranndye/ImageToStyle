from fastapi import FastAPI
import pathlib
from pydantic import BaseModel


DATAPATH = pathlib.Path(__file__).parent / "data"


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


class User_input(BaseModel):
    id_number: str
    prompt: str


class User_input2(BaseModel):
    number_1: str
    number_2: str


@app.post("/items/")
def create_item(values: User_input):
    return values


@app.post("/add")
def add_numbers(values: User_input2) -> str:
    value_1 = float(values.number_1)
    value_2 = float(values.number_2)
    result = str(value_1 + value_2)
    return result
