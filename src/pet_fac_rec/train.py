from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from model import MyEfficientNetModel
from torch.utils.data import DataLoader
from torchvision import transforms

from data import MyDataset, get_default_transforms

# Determine the device to use for training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(DEVICE)


def train(
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 1,
    data_csv: Path = Path("data/data.csv"),
    num_classes: int = 10,
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
    print("Training with custom dataset")
    print(f"{lr=}, {batch_size=}, {epochs=}, {num_classes=}, {data_csv=}")

    # Initialize the model
    model = MyEfficientNetModel(num_classes=num_classes).to(DEVICE)

    # Load the dataset
    transform = get_default_transforms()
    dataset = MyDataset(csv_file=data_csv, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Statistics for visualization
    statistics = {"train_loss": [], "train_accuracy": []}

    # Training loop
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)

            # Forward pass
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Record statistics
            statistics["train_loss"].append(loss.item())
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}, accuracy: {accuracy:.4f}")

    print("Training complete")

    # Save the model
    torch.save(model.state_dict(), "models/model.pth")

    # Plot training statistics
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")


# Entry point for running the script
def main() -> None:
    """
    Main entry point for the training script.
    """
    typer.run(train)


if __name__ == "__main__":
    main()
