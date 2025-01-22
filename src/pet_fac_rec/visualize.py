import logging
from datetime import datetime

import matplotlib.pyplot as plt


current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(filename=f"reports/logs/{current_time}.log", level=logging.INFO)
log = logging.getLogger(__name__)


def plot_training_statistics(
    train_losses, train_accuracies, val_losses, val_accuracies, output_path="reports/figures/training_statistics.png"
):
    """
    Plot training and validation statistics.

    Args:
        train_losses (list): Training losses over epochs.
        train_accuracies (list): Training accuracies over epochs.
        val_losses (list): Validation losses over epochs.
        val_accuracies (list): Validation accuracies over epochs.
        output_path (str): Path to save the plot image.
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot Training Loss
    axs[0][0].plot(train_losses, label="Train Loss")
    axs[0][0].set_title("Train Loss")
    axs[0][0].set_xlabel("Epochs")
    axs[0][0].set_ylabel("Loss")
    axs[0][0].legend()

    # Plot Training Accuracy
    axs[0][1].plot(train_accuracies, label="Train Accuracy")
    axs[0][1].set_title("Train Accuracy")
    axs[0][1].set_xlabel("Epochs")
    axs[0][1].set_ylabel("Accuracy")
    axs[0][1].legend()

    # Plot Validation Loss
    axs[1][0].plot(val_losses, label="Validation Loss")
    axs[1][0].set_title("Validation Loss")
    axs[1][0].set_xlabel("Epochs")
    axs[1][0].set_ylabel("Loss")
    axs[1][0].legend()

    # Plot Validation Accuracy
    axs[1][1].plot(val_accuracies, label="Validation Accuracy")
    axs[1][1].set_title("Validation Accuracy")
    axs[1][1].set_xlabel("Epochs")
    axs[1][1].set_ylabel("Accuracy")
    axs[1][1].legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    fig.savefig(output_path)
    log.info(f"Training statistics saved to {output_path}")
