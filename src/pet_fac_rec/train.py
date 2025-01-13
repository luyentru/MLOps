import matplotlib.pyplot as plt
import torch
import typer
import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from pet_fac_rec.model import MyEfficientNetModel, MyResNet50Model, MyVGG16Model
from pet_fac_rec.data import MyDataset, get_default_transforms
from tqdm import tqdm

app = typer.Typer()


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
    lr: float = 1e-3,
    batch_size: int = 8,
    epochs: int = 2,
    data_csv: Path = Path("data/data.csv"),
    seed: int = 42,
) -> None:
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
    set_seed(seed)

    # Determine the device to use for training
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Running on dev: {device}")

    print("Training parameters:")
    print(f"{lr=}, {batch_size=}, {epochs=}, {data_csv=}")

    # Load the dataset
    transform = get_default_transforms()
    train_dataset = MyDataset(csv_file=data_csv, split="train", transform=transform)
    valid_dataset = MyDataset(csv_file=data_csv, split="valid", transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    num_classes = train_dataset.num_classes
    model = get_model(model_name, num_classes).to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    # Training loop
    epoch_bar = tqdm(range(epochs))
    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        total_correct = 0
        total_samples = 0

        for img, target in iter(train_dataloader):
            img, target = img.to(device), target.to(device)

            # Forward pass
            y_pred = model(img)
            loss = criterion(y_pred, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
        epoch_bar.set_description(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_losses[-1]:.5f} | Train Acc: {train_accuracies[-1]:.5f} | Val Loss: {val_losses[-1]:.5f} | Val Acc: {val_accuracies[-1]:.5f}"
        )

    print("Training complete")

    # Save the model
    # Save the model
    model_save_path = f"models/{model_name}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


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
