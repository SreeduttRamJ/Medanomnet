import streamlit as st
from PIL import Image
import os
from utils.inference import process_image

st.set_page_config(page_title="MedAnomNet - Anomaly Detection", layout="centered")

st.title("🧠 MedAnomNet")
st.markdown("Upload an image (Brain MRI, Liver CT, or Chest X-ray) for anomaly detection and visualization.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    os.makedirs("static/uploads", exist_ok=True)
    filepath = os.path.join("static/uploads", uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(filepath, caption="Uploaded Image", use_column_width=True)

    result = process_image(filepath)

    st.subheader("🔍 Analysis Result")
    st.write(f"**Classification:** {result['classification']}")
    st.write(f"**Confidence:** {result['confidence']}%")
    st.write(f"**Anomaly Detected:** {'✅ Yes' if result['anomaly_detected'] else '❌ No'}")

    if result['anomaly_detected']:
        st.image(result['heatmap_path'], caption="🔥 Anomaly Heatmap", use_column_width=True)

# Optional: Contact form (replaces your /submit route)
with st.expander("📨 Contact Us"):
    with st.form("contact_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Message")
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.success(f"Thanks {name}, your message has been received.")
