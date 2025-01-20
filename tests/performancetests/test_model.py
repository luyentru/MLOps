import os
import time

import torch
from dotenv import load_dotenv

import wandb
from pet_fac_rec.model import MyEfficientNetModel


load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_ENTITY_ORG = os.getenv("WANDB_ENTITY_ORG")
WANDB_REGISTRY = os.getenv("WANDB_REGISTRY")
WANDB_COLLECTION = os.getenv("WANDB_COLLECTION")
wandb.login(key=WANDB_API_KEY)


def test_model_speed():
    model_name = "efficientnet"  # os.getenv("MODEL_NAME") # TODO: Redo this
    run = wandb.init()
    artifact = run.use_artifact(f"{WANDB_ENTITY_ORG}/{WANDB_REGISTRY}/{WANDB_COLLECTION}:latest", type="model")
    artifact_dir = artifact.download()
    model = MyEfficientNetModel(num_classes=4)
    model.load_state_dict(torch.load(f"{artifact_dir}/{model_name}.pth"))

    model.eval()

    # Generate a random input appropriate for your model
    test_input = torch.rand(1, 3, 224, 224)  # Example for an image model

    # Timing the model predictions
    start_time = time.time()
    num_predictions = 1
    for _ in range(num_predictions):
        model(test_input)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time for {num_predictions} predictions: {total_time:.2f} seconds")

    # Assert that 100 predictions are made in less than X seconds, adjust X as needed
    assert total_time < 10, "Model predictions are too slow"


if __name__ == "__main__":
    test_model_speed()
