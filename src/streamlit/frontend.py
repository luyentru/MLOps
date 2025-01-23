import requests
from google.cloud import run_v2

import streamlit as st


def get_backend_url():
    """
    Get the URL of the backend service.
    Works for Google Cloud setup (if backend container is running) and locally (via backend:5000)
    """

    try:
        parent = "projects/pet-fac-rec/locations/europe-west1"
        client = run_v2.ServicesClient()
        services = client.list_services(parent=parent)
        for service in services:
            if service.name.split("/")[-1] == "backend-bento":
                return service.uri
    except Exception:
        print("container cannot find backend on gcloud, trying local docker network (backend:5000)")
        return "http://backend:5000"


def classify_image(image, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/predict"
    files = {"file": ("image.jpg", image, "image/jpeg")}
    headers = {"accept": "application/json"}
    response = requests.post(predict_url, headers=headers, files=files, timeout=100)
    if response.status_code == 200:
        return response.json()
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Pet's Facial Expression Classifier")
    st.subheader("Group Project (Group 65) of MLOps Class at DTU")
    st.write("""
        This application allows you to upload an image of your pet and receive a classification
        of its facial expression. The model has been trained on cats, dogs, horses, and hamsters and is able to
        differentiate between the 4 classes 'sad', 'happy', 'angry', 'other'.""")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = uploaded_file.read()
        result = classify_image(image, backend=backend)

        if result is not None:
            class_name = result["class_name"]
            probability = result["probability"]

            # show the image and prediction
            st.image(image, caption="Uploaded Image")
            st.write("Prediction:", class_name)
            st.write("Probability:", probability)
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()
