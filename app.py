# app.py
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16
    ).to("cuda")
    return pipe

st.title("DreamHome AI - Generate House Designs")
prompt = st.text_input("Enter prompt (e.g., 2BHK floor plan, modern house)")

if st.button("Generate"):
    if prompt:
        pipe = load_model()
        with st.spinner("Generating..."):
            image = pipe(prompt).images[0]
        st.image(image, caption="Generated Design", use_column_width=True)
    else:
        st.warning("Please enter a prompt.")
