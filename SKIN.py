import streamlit as st
from PIL import Image
import requests, base64, io

# UI Title
st.title("🌿 Skin Disease Detection and Routine Recommendation System")

# Image Upload
uploaded_file = st.file_uploader("Upload the picture of affected area", type=["jpg", "jpeg", "png"])

# Compress and Encode
def compress_image_to_fit(file, max_b64_len=180_000, resize_max=512):
    img = Image.open(file).convert("RGB")
    img.thumbnail((resize_max, resize_max))
    buffer = io.BytesIO()
    quality = 95

    while quality >= 10:
        buffer.seek(0)
        buffer.truncate()
        img.save(buffer, format="JPEG", quality=quality)
        b64 = base64.b64encode(buffer.getvalue()).decode()
        if len(b64) < max_b64_len:
            return b64
        quality -= 5

    raise ValueError("❌ Could not compress image within required size.")

# When an image is uploaded
if uploaded_file:
    try:
        image_b64 = compress_image_to_fit(uploaded_file)
        st.success("✅ Image successfully compressed!")

        # Prompt template — adjusted for skin disease
        prompt = f"""Analyze this skin image strictly in this format:
Detected Disease: [name]
Causes: [causes]
Recommended drugs: [drugs]
Natural treatments: [natural options]
Specific diet changes: [diet]
Routine changes: [routine]
Best skincare products with online purchase links: [products]
<img src="data:image/jpeg;base64,{image_b64}" />
"""

        # API Info — your skin disease API key & endpoint
        invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        api_key = "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"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
        payload = {
            "model": "google/gemma-3-27b-it",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.2,
            "top_p": 0.7,
            "stream": False
        }

        # Analyze Button
        if st.button("🔍 Analyze Skin Image"):
            with st.spinner("Analyzing"):
                response = requests.post(invoke_url, headers=headers, json=payload)
                result = response.json()
                message = result["choices"][0]["message"]["content"]
                st.subheader("📋 Diagnosis & Recommendations")
                formatted_message = message.replace("Detected Disease:", "\n\n**Detected Disease:**") \
                    .replace("Causes:", "\n\n**Causes:**") \
                    .replace("Recommended drugs:", "\n\n**Recommended Drugs:**") \
                    .replace("Natural treatments:", "\n\n**Natural Treatments:**") \
                    .replace("Specific diet changes:", "\n\n**Specific Diet Changes:**") \
                    .replace("Routine changes:", "\n\n**Routine Changes:**") \
                    .replace("Best skincare products with online purchase links:", "\n\n**Best Skincare Products:**")
                st.markdown(formatted_message)

    except Exception as e:
        st.error(str(e))
