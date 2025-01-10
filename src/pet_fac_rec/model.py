import torch
from torch import nn
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
from typing import Tuple

class MyEfficientNetModel(nn.Module):
    """
    A model leveraging EfficientNet as the base feature extractor.

    Attributes:
        base_model (nn.Module): Pretrained EfficientNet model.
        fc_layers (nn.Sequential): Fully connected layers for classification.
    """

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        """
        Initialize the model with EfficientNet as the backbone.

        Args:
            num_classes (int): Number of output classes for classification.
            pretrained (bool): Whether to use a pretrained EfficientNet model.
        """
        super().__init__()

        # Load EfficientNet as the base model with updated weights argument
        weights = EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None
        self.base_model = efficientnet_b5(weights=weights)

        # Remove the final classification layer
        self.base_model.classifier = nn.Identity()

        # Define additional classification layers
        self.fc_layers = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = self.base_model(x)
        x = self.fc_layers(x)
        return x

if __name__ == "__main__":
    # Example usage
    num_classes = 10  # Replace with the actual number of classes
    model = MyEfficientNetModel(num_classes=num_classes)

    # Print model architecture and parameter count
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Test the model with a dummy input
    dummy_input = torch.randn(1, 3, 224, 224)  # EfficientNet requires 3-channel 224x224 input
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
