from typing import Tuple

import torch
from torch import nn
from torchvision.models import efficientnet_b5, resnet50, vgg16


class MyEfficientNetModel(nn.Module):
    """
    A model leveraging EfficientNet as the base feature extractor.
    """

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        self.base_model = efficientnet_b5(pretrained=pretrained)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(nn.Linear(num_features, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


class MyResNet50Model(nn.Module):
    """
    A model leveraging ResNet50 as the base feature extractor.
    """

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        self.base_model = resnet50(pretrained=pretrained)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


class MyVGG16Model(nn.Module):
    """
    A model leveraging VGG16 as the base feature extractor.
    """

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        self.base_model = vgg16(pretrained=pretrained)
        num_features = self.base_model.classifier[6].in_features
        self.base_model.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)
