import streamlit as st
import requests

st.title("MNIST Digit Classifier")

uploaded_file = st.file_uploader(
    "Upload Digit Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image")

    if st.button("Predict"):
        files = {
            "file": uploaded_file.getvalue()
        }

        response = requests.post(
            "https://mnistproject-production.up.railway.app/predict",
            files=files
        )

        st.success(f"Predicted Digit: {response.json()['prediction']}")