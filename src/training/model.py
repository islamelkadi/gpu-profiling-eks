"""ResNet-18 model adapted for CIFAR-10."""

import logging

import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)


def build_model(args, device):
    """Build ResNet-18 adapted for CIFAR-10 (10 classes, 32x32 input)."""
    model = models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)

    if getattr(args, "compile", False):
        if hasattr(torch, "compile"):
            logger.info("Applying torch.compile to model")
            model = torch.compile(model)
        else:
            logger.warning("torch.compile not available, skipping")

    return model
