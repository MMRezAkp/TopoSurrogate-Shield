#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model definitions for the neural activation extractor.

This module provides a unified interface for loading different neural network
architectures used in the activation extraction and topological analysis.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_model(num_classes=10, pretrained=False, architecture="resnet18"):
    """
    Get a model instance based on the specified architecture.
    
    Args:
        num_classes (int): Number of output classes (default: 10 for CIFAR-10)
        pretrained (bool): Whether to use pretrained weights (default: False)
        architecture (str): Model architecture name
        
    Returns:
        torch.nn.Module: Model instance
        
    Supported architectures:
        - resnet18, resnet34, resnet152
        - mobilenet_v2
        - efficientnet_b0
    """
    
    if architecture == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        # Modify final layer for the target number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif architecture == "resnet34":
        model = models.resnet34(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif architecture == "resnet152":
        model = models.resnet152(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif architecture == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
        # MobileNetV2 has a classifier instead of fc
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        
    elif architecture == "efficientnet_b0":
        # Note: EfficientNet might require timm library for best support
        # For basic functionality, we can use a simple implementation
        try:
            import timm
            model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)
        except ImportError:
            # Fallback to torchvision if timm is not available
            print("Warning: timm not available. Using torchvision EfficientNet implementation.")
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(model.classifier[1].in_features, num_classes)
            )
    
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    return model


def get_supported_architectures():
    """
    Get list of supported architectures.
    
    Returns:
        list: List of supported architecture names
    """
    return ["resnet18", "resnet34", "resnet152", "mobilenet_v2", "efficientnet_b0"]


def print_model_info(model, architecture):
    """
    Print basic information about the model.
    
    Args:
        model (torch.nn.Module): Model instance
        architecture (str): Architecture name
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Architecture: {architecture}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size (MB): {total_params * 4 / (1024 * 1024):.2f}")


if __name__ == "__main__":
    # Example usage and testing
    print("Testing model creation...")
    
    for arch in get_supported_architectures():
        try:
            model = get_model(num_classes=10, pretrained=False, architecture=arch)
            print_model_info(model, arch)
            print("-" * 50)
        except Exception as e:
            print(f"Error creating {arch}: {e}")
            print("-" * 50)