import streamlit as st

from PIL import Image


from styletransfer import (
    SAMPLE_PATH,
    STYLE_PATH,
    CONTENT_PATH,
    image_trainer,
)


cont_img = Image.open(CONTENT_PATH)
style_img = Image.open(STYLE_PATH)
sample_image = Image.open(SAMPLE_PATH)


st.set_page_config(
    page_title="Neural Style Transfer",
    initial_sidebar_state="expanded",
)

st.sidebar.markdown(" ## Build Your Own")


prog_intro = (
    "This program allows you"
    " to build your own neural style"
    " transfer model. The program is"
    " based on the paper"
    " https://arxiv.org/pdf/1508.06576.pdf, "
    "PyTorch's neural style"
    " tutorial, and "
    "https://towardsdatascience.com"
    "/implementing-neural-style-trans"
    "fer-using-pytorch-fd8d43fb7bfa."
)
more_info = (
    "This program uses the "
    "pretrained VGG model weights. "
    "Selected layers of this model are "
    "used to construct the neural transfer "
    "model. Select layers to use with "
    "the content and select layers "
    "to use with the style. "
    "Then choose relative weights "
    "for the content and style layers."
    "Push the Style transfer button "
    "to see how your model works."
    "The sample image uses the layers "
    "and recommended in the paper."
)


st.sidebar.write(prog_intro)


st.sidebar.info(
    "More about art and quilting at"
    " [www.heatheranndye.com](www.heatheranndye.com)"
    "More information about machine learning at"
    " [my github blog](https://heatheranndye.github.io/) ",
    icon="ℹ️",
)

st.title("Neural Style Transfer Modeler")
with st.container():
    st.write("Sample Neural Transfer")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Content Image")
        st.image(cont_img, width=200)
    with col2:
        st.subheader("Style Image")
        st.image(style_img, width=200)
    with col3:
        st.subheader("Generated Image")
        st.image(sample_image, width=200)

st.write("Select the layers to build your model")

options_content = st.multiselect("Content Layers", [1, 2, 3, 4, 5])
options_style = st.multiselect("Style Layers", [1, 2, 3, 4, 5])
alpha = st.slider("Weight of Content", min_value=1, max_value=100)
beta = st.slider("Weight of the Style", min_value=1, max_value=20)
if st.button("Style Transfer:"):
    result = image_trainer(
        alpha=alpha,
        beta=beta,
        content_layers=options_content,
        style_layers=options_style,
    )
    st.image(result)
