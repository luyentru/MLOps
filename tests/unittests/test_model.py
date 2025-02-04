import torch

from pet_fac_rec.model import MyEfficientNetModel


def test_efficientnet_model_init():
    model = MyEfficientNetModel(num_classes=4)
    # Check if the final layer has correct output dimensions
    assert model.base_model.classifier[-1].out_features == 4, "EfficientNet does not create 4 output classes"

def test_efficientnet_forward_pass():
    model = MyEfficientNetModel(num_classes=4)
    # Create sample input (batch_size=1, channels=3, height=224, width=224)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)

    # Check output shape and type
    assert output.shape == (2, 4), "EfficientNet does not create 4 output classes"
    assert isinstance(output, torch.Tensor), "Model output is not a PyTorch tensor"


def test_efficientnet_pretrained_parameter():
    # Test both pretrained options
    model_pretrained = MyEfficientNetModel(num_classes=4, pretrained=True)
    model_not_pretrained = MyEfficientNetModel(num_classes=4, pretrained=False)

    assert isinstance(model_pretrained, MyEfficientNetModel), (
        "Pretrained model is not an instance of MyEfficientNetModel"
    )
    assert isinstance(model_not_pretrained, MyEfficientNetModel), (
        "Non-pretrained model is not an instance of MyEfficientNetModel"
    )
