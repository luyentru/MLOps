import torch
import torch.nn as nn
from torchvision.models import efficientnet_b5


class MyEfficientNetModel(nn.Module):
    """
    Mimics the real EfficientNet-based model structure with a dummy feature extractor.
    """

    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        self.base_model = efficientnet_b5(pretrained=pretrained)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(nn.Linear(num_features, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


# Generate a dummy state_dict
def generate_dummy_model(num_classes: int, model_path: str):
    model = MyEfficientNetModel(num_classes=num_classes)
    for param in model.parameters():
        param.data = torch.randn_like(param)

    torch.save(model.state_dict(), model_path)
    print(f"Dummy model saved to {model_path}")


if __name__ == "__main__":
    NUM_CLASSES = 4
    MODEL_PATH = "models/efficientnet.pth"
    generate_dummy_model(NUM_CLASSES, MODEL_PATH)
