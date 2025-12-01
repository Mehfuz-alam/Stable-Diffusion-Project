import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import os

st.set_page_config(page_title="Stable Diffusion XL Generator", layout="wide")

st.title("🌌 Stable Diffusion XL Image Generator")
st.write("Generate AI images from text prompts using Hugging Face's Stable Diffusion XL model!")

# Sidebar for user input
with st.sidebar:
    st.header("Settings")
    prompt = st.text_area("Enter your prompt:", "An astronaut riding a green horse")
    width = st.slider("Width", 256, 1024, 512, step=64)
    height = st.slider("Height", 256, 1024, 512, step=64)
    num_images = st.slider("Number of images", 1, 4, 1)
    st.write("---")
    st.write("💡 Tip: Keep prompt descriptive for better results!")

# Hugging Face API
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HF_API_TOKEN = st.secrets.get("HF_API_TOKEN")  
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def generate_image(prompt, width=512, height=512):
    payload = {
        "inputs": prompt,
        "options": {"wait_for_model": True},
        "parameters": {"width": width, "height": height, "num_inference_steps": 30}
    }
    response = requests.post(HF_API_URL, headers=headers, json=payload)

    try:
        return Image.open(BytesIO(response.content))
    except Exception:
        st.error(f"Error generating image: {response.text}")
        return None

# Generate images button
if st.button("Generate Image(s)"):
    if not prompt.strip():
        st.warning("Please enter a prompt!")
    else:
        st.info("Generating image(s)... This may take a few seconds.")
        images = []
        for i in range(num_images):
            img = generate_image(prompt, width, height)
            if img:
                images.append(img)
        
        if images:
            st.success(f"Generated {len(images)} image(s)!")
            cols = st.columns(len(images))
            for idx, img in enumerate(images):
                with cols[idx]:
                    st.image(img, caption=f"Image {idx+1}", use_column_width=True)
                    # Convert image to bytes for download
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.download_button(
                        label="Download PNG",
                        data=byte_im,
                        file_name=f"image_{idx+1}.png",
                        mime="image/png"
                    )
