import logging
from typing import List, Optional
import random
from datetime import datetime

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import torch
from torch.profiler import record_function
import typer
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from pet_fac_rec.model import MyEfficientNetModel, MyResNet50Model, MyVGG16Model
from pet_fac_rec.data import MyDataset, get_default_transforms
from tqdm import tqdm
import onnx

app = typer.Typer()

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(filename=f"reports/logs/{current_time}.log", level=logging.INFO)
log = logging.getLogger(__name__)


def my_compose(overrides: Optional[List[str]]) -> DictConfig:
    with initialize(config_path="configs", job_name="train_model"):
        return compose(config_name="config.yaml", overrides=overrides)


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # For CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_name: str, num_classes: int) -> torch.nn.Module:
    """
    Returns the model based on the model name.
    """
    if model_name == "efficientnet":
        return MyEfficientNetModel(num_classes=num_classes)
    elif model_name == "resnet50":
        return MyResNet50Model(num_classes=num_classes)
    elif model_name == "vgg16":
        return MyVGG16Model(num_classes=num_classes)
    else:
        raise ValueError("Unsupported model type provided!")


@app.command()
def train(
    model_name: str = typer.Option("efficientnet", help="Model type to use ('efficientnet', 'resnet50', 'vgg16')"),
    overrides: Optional[List[str]] = typer.Argument(None),
) -> None:
    cfg = my_compose(overrides)
    print(f"Configuration: {OmegaConf.to_yaml(cfg)}")  # Remove later
    log.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    hparams = cfg.experiment
    """
    Train the MyEfficientNetModel on the custom dataset.

    Args:
        lr (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train the model.
        data_csv (Path): Path to the CSV file containing the preprocessed data.
        num_classes (int): Number of output classes in the dataset.
    """
    # Set the seed for reproducibility
    set_seed(hparams.seed)

    # Determine the device to use for training
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Running on dev: {device}")  # Remove later

    # Load the dataset
    transform = get_default_transforms()
    train_dataset = MyDataset(csv_file=Path(hparams.data_csv), split="train", transform=transform)
    valid_dataset = MyDataset(csv_file=Path(hparams.data_csv), split="valid", transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)
    val_dataloader = DataLoader(valid_dataset, batch_size=hparams.batch_size, shuffle=True)

    # Initialize the model
    num_classes = train_dataset.num_classes
    model = get_model(model_name, num_classes).to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr)

    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    # Training loop
    log.info(f"Start training {model_name}...")
    epoch_bar = tqdm(range(hparams.epochs))
    for epoch in epoch_bar:
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("profiler"),
            with_stack=True,
        ) as profiler:
            model.train()
            train_loss = 0.0
            total_correct = 0
            total_samples = 0

            for img, target in iter(train_dataloader):
                img, target = img.to(device), target.to(device)

                # Forward pass
                y_pred = model(img)
                with record_function("model_loss"):
                    loss = criterion(y_pred, target)

                # Backward pass and optimization
                optimizer.zero_grad()
                with record_function("backward"):
                    loss.backward()
                    optimizer.step()
                profiler.step()

                train_loss += loss.item() * img.size(0)
                _, preds = torch.max(y_pred, 1)
                total_correct += (preds == target).sum().item()
                total_samples += target.size(0)

            train_losses.append(train_loss / total_samples)
            train_accuracies.append(total_correct / total_samples)

            val_loss = 0.0
            total_correct = 0
            total_samples = 0

            model.eval()

            with torch.no_grad():
                for img, target in iter(val_dataloader):
                    img, target = img.to(device), target.to(device)

                    outputs = model(img)
                    loss = criterion(outputs, target)

                    val_loss += loss.item() * img.size(0)
                    _, preds = torch.max(outputs, 1)
                    total_correct += (preds == target).sum().item()
                    total_samples += target.size(0)

            val_losses.append(val_loss / total_samples)
            val_accuracies.append(total_correct / total_samples)
            status = f"Epoch {epoch + 1}/{hparams.epochs} | Train Loss: {train_losses[-1]:.5f} | Train Acc: {train_accuracies[-1]:.5f} | Val Loss: {val_losses[-1]:.5f} | Val Acc: {val_accuracies[-1]:.5f}"
            epoch_bar.set_description(status)
            log.info(status)

    log.info("Training complete")

    # Save the model
    model_save_path = f"models/{model_name}.pth"
    torch.save(model.state_dict(), model_save_path)
    log.info(f"Model saved to {model_save_path}")

    # Export ONNX file
    try:
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        onnx_save_path = f"models/{model_name}_v2.onnx"

        # Export the model
        torch.onnx.export(
            model,
            dummy_input,
            onnx_save_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11,  # Specify ONNX opset version
            do_constant_folding=True,  # Optimize constant-folding
            export_params=True,  # Store the trained parameter weights inside the model file
        )

        # Verify the model
        onnx_model = onnx.load(onnx_save_path)
        onnx.checker.check_model(onnx_model)

        # Add model metadata
        onnx_model.graph.doc_string = f"Pet facial recognition model using {model_name}"
        onnx.save(onnx_model, onnx_save_path)

        log.info(f"Model exported and verified to ONNX at {onnx_save_path}")
    except Exception as e:
        log.error(f"Failed to export ONNX model: {str(e)}")

    # TODO: Make a seperate plotting function
    # Plot training statistics
    fig, axs = plt.subplots(2, 2, figsize=(15, 5))
    axs[0][0].plot(train_losses)
    axs[0][0].set_title("Train loss")
    axs[0][1].plot(train_accuracies)
    axs[0][1].set_title("Train accuracy")
    axs[1][0].plot(val_losses)
    axs[1][0].set_title("Validation loss")
    axs[1][1].plot(val_accuracies)
    axs[1][1].set_title("Validation accuracy")
    fig.savefig("reports/figures/training_statistics.png")


if __name__ == "__main__":
    app()
