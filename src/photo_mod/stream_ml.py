import streamlit as st
import pathlib
from PIL import Image
import json
import requests

DATAPATH = pathlib.Path(__file__).parent / "data"

st.title("Image to Image Diffusion")

st.write("Select a base image")

option = st.selectbox("Choose a number", ("1", "2", "3", "4"))


image_name = "flower" + str(option) + ".jpg"
image_file = DATAPATH / image_name

st.write(image_file)

image = Image.open(image_file)
st.image(image)


user_prompt = st.text_input(label="Add a prompt")

inputs = {"id_number": option, "prompt": user_prompt}

if st.button("Modify:"):
    res = requests.post(
        url="http://127.0.0.1:8000/generate",
        data=json.dumps(inputs),
    )

response_image_file = DATAPATH / "image.png"
st.image(response_image_file)
