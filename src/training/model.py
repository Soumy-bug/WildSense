import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def build_model(num_classes, freeze_backbone=True, pretrained=True):
    """
    Builds a ResNet50 and replaces its final classification layer to match
    our number of species.

    pretrained=True (default, used for training): loads ImageNet-pretrained
    weights as a starting point — this is what transfer learning needs.

    pretrained=False (used for inference): skips downloading ImageNet
    weights entirely, since we're about to load our own fully-trained
    checkpoint anyway and would just overwrite them immediately. This
    matters a lot on memory-constrained deployments (e.g. free-tier
    hosting) — downloading and holding an extra ~98MB of weights we're
    about to discard was enough to cause an out-of-memory crash.

    freeze_backbone=True freezes every layer except the new final layer,
    so early training only updates the new classifier head. Call
    unfreeze_backbone() later to fine-tune the whole network.
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def unfreeze_backbone(model):
    """
    Unfreezes all layers so the whole network can fine-tune. Call this
    after the classifier head has trained for a few epochs, then continue
    training with a much lower learning rate (see train.py CONFIG).
    """
    for param in model.parameters():
        param.requires_grad = True
    return model