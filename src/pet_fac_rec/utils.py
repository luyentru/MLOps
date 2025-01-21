import matplotlib.pyplot as plt
import numpy as np


def show_image_and_target(images, targets, show=True):
    """
    Display a grid of images with their corresponding target labels.

    Args:
        images (list or torch.Tensor): A list or tensor of images to display.
        targets (list or torch.Tensor): A list or tensor of target labels.
        show (bool): Whether to display the plot immediately. If False, the plot can be saved.
    """
    num_images = len(images)
    grid_size = int(np.ceil(np.sqrt(num_images)))  # Determine grid dimensions
    plt.figure(figsize=(10, 10))

    for i in range(num_images):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")  # Squeeze in case of single-channel images
        plt.title(f"Label: {targets[i]}")
        plt.axis("off")  # Hide axis ticks for cleaner visuals

    plt.tight_layout()

    if show:
        plt.show()
