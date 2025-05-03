import streamlit as st
from PIL import Image
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline  # updated import
import torch

# Optional debug check
st.write("CUDA available:", torch.cuda.is_available())

# Load the Stable Diffusion model
@st.cache_resource
def load_model():
    model = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        revision="fp16"
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    return model

pipe = load_model()

# Streamlit UI
st.title("DreamHome AI")
prompt = st.text_input("Enter your home design prompt:")

if st.button("Generate"):
    if prompt:
        with st.spinner("Generating image..."):
            image = pipe(prompt).images[0]
            st.image(image, caption="Generated Design", use_column_width=True)
