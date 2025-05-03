import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Title
st.title("DreamHome AI - Text to Floor Plan Generator")

# Load model once
@st.cache_resource
def load_model():
    model = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model

pipe = load_model()

# Prompt input
prompt = st.text_input("Enter your prompt (e.g., modern 2-bedroom house plan):")

# Generate on click
if st.button("Generate Floor Plan") and prompt:
    with st.spinner("Generating..."):
        image = pipe(prompt).images[0]
        st.image(image, caption="Generated Floor Plan", use_column_width=True)
