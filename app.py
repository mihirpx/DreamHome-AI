# app.py
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

st.set_page_config(page_title="DreamHome AI", layout="centered")

@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_pipeline()

st.title("🏡 DreamHome AI - Floorplan and Design Generator")
prompt = st.text_input("Enter a prompt (e.g. 'Floor plan for a 2BHK modern house')")

if st.button("Generate Image") and prompt:
    with st.spinner("Generating..."):
        result = pipe(prompt).images[0]
        st.image(result, caption="Generated Image", use_column_width=True)
