import streamlit as st
import json
import requests
import pathlib
from ml_imagetoimage import DATAPATH

st.title("Image to Image Diffusion")

st.write("Select a base image")

option = st.selectbox("Choose a number", ("1", "2", "3", "4"))


image_file = "data/flower" + str(option) + ".jpg"
st.write(image_file)

st.image(image_file)


user_prompt = st.text_input(label="Add a prompt")

inputs = {"id_number": option, "prompt": user_prompt}

if st.button("Modify:"):
    res = requests.post(
        url="http://127.0.0.1:8000/generate",
        data=json.dumps(inputs),
    )

st.image("image.png")
