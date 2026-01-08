"""
Construção de modelos usando transfer learning.
Carrega modelos pré-treinados do torchvision e adapta para classificação.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

from ..domain.models import ModelType


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Constrói um modelo de classificação usando transfer learning.
    
    Args:
        model_name: Nome do modelo (MobileNetV2, VGG16 ou VGG19)
        num_classes: Número de classes para classificação
        pretrained: Se True, usa pesos pré-treinados no ImageNet
        
    Returns:
        Modelo PyTorch configurado
    """
    model_name_upper = model_name.upper()
    
    if model_name_upper == ModelType.MOBILENETV2.value.upper():
        return _build_mobilenetv2(num_classes, pretrained)
    elif model_name_upper == ModelType.VGG16.value.upper():
        return _build_vgg16(num_classes, pretrained)
    elif model_name_upper == ModelType.VGG19.value.upper():
        return _build_vgg19(num_classes, pretrained)
    else:
        raise ValueError(
            f"Modelo '{model_name}' não suportado. "
            f"Modelos disponíveis: {ModelType.MOBILENETV2.value}, {ModelType.VGG16.value}, {ModelType.VGG19.value}"
        )


def _build_mobilenetv2(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Constrói MobileNetV2 para classificação.
    
    Args:
        num_classes: Número de classes
        pretrained: Se True, usa pesos pré-treinados
        
    Returns:
        Modelo MobileNetV2 configurado
    """
    # Carrega modelo pré-treinado
    model = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
    
    # Congela todas as camadas exceto o classificador
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Substitui o classificador
    # MobileNetV2 tem um classificador com 1280 features de entrada
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    
    return model


def _build_vgg16(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Constrói VGG16 para classificação.
    
    Args:
        num_classes: Número de classes
        pretrained: Se True, usa pesos pré-treinados
        
    Returns:
        Modelo VGG16 configurado
    """
    # Carrega modelo pré-treinado
    model = models.vgg16(weights='IMAGENET1K_V1' if pretrained else None)
    
    # Congela todas as camadas de features (convolutional layers)
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Substitui o classificador
    # VGG16 tem um classificador com 25088 features de entrada (após flatten)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    
    return model


def _build_vgg19(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Constrói VGG19 para classificação.
    
    Args:
        num_classes: Número de classes
        pretrained: Se True, usa pesos pré-treinados
        
    Returns:
        Modelo VGG19 configurado
    """
    # Carrega modelo pré-treinado
    model = models.vgg19(weights='IMAGENET1K_V1' if pretrained else None)
    
    # Congela todas as camadas de features (convolutional layers)
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Substitui o classificador
    # VGG19 tem um classificador com 25088 features de entrada (após flatten)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    
    return model
