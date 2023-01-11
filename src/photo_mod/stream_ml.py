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


user_prompt1 = st.number_input(label="value1")
user_prompt2 = st.number_input(label="value2")


inputs = {"number_1": user_prompt1, "number_2": user_prompt2}

if st.button("Modify:"):
    res = requests.post(
        url="http://127.0.0.1:8000/add",
        data=json.dumps(inputs),
    )
    st.write(res.text)
