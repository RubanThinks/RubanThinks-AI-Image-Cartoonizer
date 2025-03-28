'''import torch
import streamlit as st
import os
import qrcode
import tempfile
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionInstructPix2PixPipeline
import requests
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Using device: {device}")
MODEL_NAME = "instruction-tuning-sd/cartoonizer"
try:
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()
st.title("📸 Live Image Cartoonizer 🎨")
captured_image = st.camera_input("Take a photo")
def upload_to_imgur(image):
    """Uploads image to Imgur and returns the public URL."""
    CLIENT_ID = "d86bc4bc36bd909"  # Replace with your Imgur client ID
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        image.save(temp_file.name, format="PNG")
        temp_file_path = temp_file.name
    headers = {"Authorization": f"Client-ID {CLIENT_ID}"}
    with open(temp_file_path, "rb") as img:
        response = requests.post(
            "https://api.imgur.com/3/upload",
            headers=headers,
            files={"image": img},
        )
    if response.status_code == 200:
        return response.json()["data"]["link"]
    else:
        st.error("Failed to upload image to Imgur. Try again.")
        return None
def generate_qr_code(url):
    """Generates a QR code for the given URL."""
    qr = qrcode.make(url)
    qr_io = BytesIO()
    qr.save(qr_io, format="PNG")
    return qr_io.getvalue()
def add_text_to_image(image):
    image = image.convert("RGB")  
    draw = ImageDraw.Draw(image)
    FONT_DIR = "fonts"
    AVS_FONT_PATH = os.path.join(FONT_DIR, "times.ttf")
    ALGOVERSE_FONT_PATH = os.path.join(FONT_DIR, "impact.ttf")
    avs_font = ImageFont.truetype(AVS_FONT_PATH, 20)
    algoverse_font = ImageFont.truetype(ALGOVERSE_FONT_PATH, 20)
    img_width, img_height = image.size
    text_avs = "AI IMAGE CARTOONIZER"
    padding = 15 
    text_bbox = draw.textbbox((0, 0), text_avs, font=avs_font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x_avs = (img_width - text_width) // 2  
    y_avs = 10  
    box_x1 = x_avs - padding
    box_y1 = y_avs - padding
    box_x2 = x_avs + text_width + padding
    box_y2 = y_avs + text_height + padding
    draw.rectangle([box_x1, box_y1, box_x2, box_y2], fill="white", outline="black", width=2)
    draw.text((x_avs, y_avs), text_avs, fill="navy", font=avs_font)
    text_algoverse = "RUBAN_THINKS"  
    text_bbox = draw.textbbox((0, 0), text_algoverse, font=algoverse_font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x_algoverse = (img_width - text_width) // 2 
    y_algoverse = img_height - text_height - 10  
    box_x1 = x_algoverse - padding
    box_y1 = y_algoverse - padding
    box_x2 = x_algoverse + text_width + padding
    box_y2 = y_algoverse + text_height + padding
    draw.rectangle([box_x1, box_y1, box_x2, box_y2], fill="white", outline="black", width=1)
    draw.text((x_algoverse, y_algoverse), text_algoverse, fill="navy", font=algoverse_font)
    return image
if captured_image:
    image = Image.open(captured_image).convert("RGB")
    if st.button("Cartoonize"):
        with st.spinner("Processing..."):
            try:
                cartoonized_image = pipeline("Cartoonize the following image", image=image).images[0]
                cartoonized_image_with_text = add_text_to_image(cartoonized_image)
                img_url = upload_to_imgur(cartoonized_image_with_text)
                if img_url:
                    qr_code_bytes = generate_qr_code(img_url)
                    st.image(cartoonized_image_with_text, caption="Cartoonized Image with Text", use_container_width=True)
                    st.image(qr_code_bytes, caption="Scan to Download", use_container_width=False)
                    st.success(f"Done! Scan the QR code or click [here]({img_url}) to download your image.")
            except Exception as e:
                st.error(f"Error during cartoonization: {e}")
'''
import torch
import streamlit as st
import os
import qrcode
import tempfile
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionInstructPix2PixPipeline
import requests

# Set device (Use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Using device: {device}")

# Load Model
MODEL_NAME = "instruction-tuning-sd/cartoonizer"
try:
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    
    # 🚀 Disable NSFW Filter (Corrected)
    def dummy_safety_checker(images, clip_input):
        return images, [False]  # Must return a list of False values

    pipeline.safety_checker = dummy_safety_checker  

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("📸 Live Image Cartoonizer 🎨")

# Webcam input
captured_image = st.camera_input("Take a photo")

# Upload to Imgur
def upload_to_imgur(image):
    """Uploads image to Imgur and returns the public URL."""
    CLIENT_ID = "d86bc4bc36bd909"  # Replace with your Imgur client ID

    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        image.save(temp_file.name, format="PNG")
        temp_file_path = temp_file.name

    headers = {"Authorization": f"Client-ID {CLIENT_ID}"}
    with open(temp_file_path, "rb") as img:
        response = requests.post(
            "https://api.imgur.com/3/upload",
            headers=headers,
            files={"image": img},
        )
    
    # Parse response
    if response.status_code == 200:
        return response.json()["data"]["link"]  # Return the public Imgur link
    else:
        st.error("Failed to upload image to Imgur. Try again.")
        return None

# Generate QR Code
def generate_qr_code(url):
    """Generates a QR code for the given URL."""
    qr = qrcode.make(url)
    qr_io = BytesIO()
    qr.save(qr_io, format="PNG")
    return qr_io.getvalue()

# Add Text to Image
def add_text_to_image(image):
    """Adds text overlay to the cartoonized image."""
    image = image.convert("RGB")  
    draw = ImageDraw.Draw(image)

    # Font Paths
    FONT_DIR = "fonts"
    AVS_FONT_PATH = os.path.join(FONT_DIR, "times.ttf")
    ALGOVERSE_FONT_PATH = os.path.join(FONT_DIR, "impact.ttf")

    try:
        avs_font = ImageFont.truetype(AVS_FONT_PATH, 20)
        algoverse_font = ImageFont.truetype(ALGOVERSE_FONT_PATH, 20)
    except IOError:
        st.error("Font files not found. Ensure the fonts exist in the 'fonts' directory.")
        return image

    # Image size
    img_width, img_height = image.size

    # **AI IMAGE CARTOONIZER (Top-Center)**
    text_avs = "AI IMAGE CARTOONIZER"
    padding = 15  
    text_bbox = draw.textbbox((0, 0), text_avs, font=avs_font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    x_avs = (img_width - text_width) // 2  
    y_avs = 10  

    box_x1 = x_avs - padding
    box_y1 = y_avs - padding
    box_x2 = x_avs + text_width + padding
    box_y2 = y_avs + text_height + padding

    draw.rectangle([box_x1, box_y1, box_x2, box_y2], fill="white", outline="black", width=2)
    draw.text((x_avs, y_avs), text_avs, fill="navy", font=avs_font)

    # **RUBAN_THINKS (Bottom-Center)**
    text_algoverse = "RUBAN_THINKS"  
    text_bbox = draw.textbbox((0, 0), text_algoverse, font=algoverse_font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    x_algoverse = (img_width - text_width) // 2  
    y_algoverse = img_height - text_height - 10  

    box_x1 = x_algoverse - padding
    box_y1 = y_algoverse - padding
    box_x2 = x_algoverse + text_width + padding
    box_y2 = y_algoverse + text_height + padding

    draw.rectangle([box_x1, box_y1, box_x2, box_y2], fill="white", outline="black", width=1)
    draw.text((x_algoverse, y_algoverse), text_algoverse, fill="navy", font=algoverse_font)

    return image

# Process Image
if captured_image:
    image = Image.open(captured_image).convert("RGB")

    if st.button("Cartoonize"):
        with st.spinner("Processing..."):
            try:
                # Apply Cartoonization
                cartoonized_image = pipeline("Cartoonize the following image", image=image).images[0]

                # Add Text to Cartoonized Image
                cartoonized_image_with_text = add_text_to_image(cartoonized_image)

                # Upload to Imgur
                img_url = upload_to_imgur(cartoonized_image_with_text)

                if img_url:
                    # Generate QR Code
                    qr_code_bytes = generate_qr_code(img_url)

                    # Display cartoonized image
                    st.image(cartoonized_image_with_text, caption="Cartoonized Image with Text", use_container_width=True)

                    # Display QR Code
                    st.image(qr_code_bytes, caption="Scan to Download", use_container_width=False)

                    st.success(f"Done! Scan the QR code or click [here]({img_url}) to download your image.")

            except Exception as e:
                st.error(f"Error during cartoonization: {e}")

