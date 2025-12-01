import torch
import torch.nn as nn
import torchvision.models as tv

def build_model(name="resnet18", num_classes=6):
    if name == "resnet18":
        model = tv.resnet18(weights=None)
        model.fc = nn.Linear(512, num_classes)
    elif name == "efficientnet_b0":
        model = tv.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(1280, num_classes)
    elif name == "vit_b_16":
        model = tv.vit_b_16(weights=None)
        model.heads[0] = nn.Linear(model.heads[0].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model {name}")
    return model
