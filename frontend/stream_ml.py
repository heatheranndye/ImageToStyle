import streamlit as st
import json
import requests

st.title("Trial")
user_prompt1 = st.number_input(label="value1")
user_prompt2 = st.number_input(label="value2")


inputs = {"number_1": user_prompt1, "number_2": user_prompt2}

if st.button("Modify:"):

    res = requests.post(
        url="http://localhost:8000/add",
        data=json.dumps(inputs),
    )
    st.write(res.text)
