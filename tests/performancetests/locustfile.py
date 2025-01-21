import random
from pathlib import Path

from locust import HttpUser
from locust import TaskSet
from locust import between
from locust import task


# Path to the directory containing test images
TEST_IMAGE_DIR = Path("tests")


class PredictTaskSet(TaskSet):
    @task
    def predict(self):
        """
        Send a POST request to the /predict/ endpoint with a test image.
        """
        # Get a random image file from the directory
        image_files = list(TEST_IMAGE_DIR.glob("*.jpg"))  # Adjust extension if necessary
        assert image_files, f"No image files found in {TEST_IMAGE_DIR}"  # Ensure directory is not empty
        test_image_path = random.choice(image_files)  # Randomly select an image

        with open(test_image_path, "rb") as image_file:
            # Send the image as part of the POST request
            files = {"file": ("test_image.jpg", image_file, "image/jpeg")}
            response = self.client.post("/predict/", files=files)

        # Log the response for debugging
        if response.status_code == 200:
            print("Prediction Response:", response.json())
        else:
            print("Failed Request:", response.status_code, response.text)


class PredictUser(HttpUser):
    tasks = [PredictTaskSet]
    wait_time = between(1, 3)  # Wait between 1 and 3 seconds between requests
