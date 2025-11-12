import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import tempfile

from scripts.detection_utils import load_model, process_camera_image
from scripts.annotation_utils import create_annotated_image
from scripts.exif_utils import extract_exif_metadata

st.set_page_config(page_title="Pinniped camera census", layout="wide")

st.title("Pinniped Camera Detection & Census Tool")

model = load_model()

location = st.text_input("Enter camera location (e.g., Nanaimo, Cowichan):")

uploaded_files = st.file_uploader(
    "Upload Camera Trap Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    results = []

    for uploaded_file in uploaded_files:
        # Read image bytes
        image = Image.open(uploaded_file)
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")

        # Extract metadata (date, time, location)
        capture_date, capture_time = extract_exif_metadata(image)

        # Run detection
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name

        detection_result = process_camera_image(tmp_file_path, model=model)

        # Display annotated image inline
        annotated_img = create_annotated_image(np.array(image), detection_result["detections"], detection_result["raw_result"])
        st.image(annotated_img, caption=f"{uploaded_file.name} — Count: {detection_result['pinniped_count']}", width='stretch')

        # Collect results for census
        results.append({
            "filename": uploaded_file.name,
            "capture_date": capture_date,
            "capture_time": capture_time,
            "location": location,
            "pinniped_count": detection_result["pinniped_count"]
        })

    # Build census DataFrame
    df = pd.DataFrame(results)
    st.subheader("📊 Pinniped Census Summary")
    st.dataframe(df, width='stretch')

    # Provide downloadable CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Download Census CSV",
        data=csv,
        file_name="pinniped_census.csv",
        mime="text/csv"
    )
