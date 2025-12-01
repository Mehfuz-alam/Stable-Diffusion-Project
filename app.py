import streamlit as st
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Stable Diffusion XL Generator", layout="wide")
st.title("🌌 Stable Diffusion XL Image Generator (Free, CPU Compatible)")

# Sidebar inputs
with st.sidebar:
    st.header("Settings")
    prompt = st.text_area("Enter your prompt:", "An astronaut riding a green horse")
    width = st.slider("Width", 256, 1024, 512, step=64)
    height = st.slider("Height", 256, 1024, 512, step=64)
    num_images = st.slider("Number of images", 1, 2, 1)  # Limit to 2 on CPU
    st.write("---")
    st.write("💡 Tip: Be descriptive!")

# Load model (cache in session to avoid reloading)
if "pipe" not in st.session_state:
    with st.spinner("Loading Stable Diffusion XL..."):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float32  # CPU compatible
        )
        pipe.to("cpu")
        st.session_state["pipe"] = pipe
else:
    pipe = st.session_state["pipe"]

# Generate images
if st.button("Generate Image(s)"):
    if not prompt.strip():
        st.warning("Please enter a prompt!")
    else:
        st.info("Generating image(s)... This may take a while on CPU!")
        images = []
        for i in range(num_images):
            image = pipe(prompt=prompt).images[0]
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
