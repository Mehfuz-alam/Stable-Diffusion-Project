import streamlit as st
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Stable Diffusion XL Generator", layout="wide")
st.title("🌌 Stable Diffusion XL Image Generator (Free, Optimized CPU)")

# Sidebar inputs
with st.sidebar:
    st.header("Settings")
    prompt = st.text_area("Enter your prompt:", "An astronaut riding a green horse")
    width = st.slider("Width", 256, 512, 512, step=64)
    height = st.slider("Height", 256, 512, 512, step=64)
    num_images = st.slider("Number of images", 1, 2, 1)  # 1-2 images on CPU
    st.write("---")
    st.write("💡 Tip: Be descriptive for better results!")

# Load model with caching
@st.cache_resource(show_spinner=True)
def load_model():
    st.info("Downloading Stable Diffusion XL model for the first time. This may take several minutes...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,  # lighter memory
        use_safetensors=True
    )
    pipe.to("cpu")  # CPU-compatible
    st.success("Model loaded! You can now generate images.")
    return pipe

pipe = load_model()

# Generate images
if st.button("Generate Image(s)"):
    if not prompt.strip():
        st.warning("Please enter a prompt!")
    else:
        st.info("Generating image(s)... This may take a minute on CPU.")
        images = []
        for i in range(num_images):
            image = pipe(prompt=prompt, width=width, height=height, num_inference_steps=20).images[0]
            images.append(image)
        
        if images:
            st.success(f"Generated {len(images)} image(s)!")
            cols = st.columns(len(images))
            for idx, img in enumerate(images):
                with cols[idx]:
                    st.image(img, caption=f"Image {idx+1}", use_column_width=True)
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    st.download_button(
                        label="Download PNG",
                        data=buf.getvalue(),
                        file_name=f"image_{idx+1}.png",
                        mime="image/png"
                    )
