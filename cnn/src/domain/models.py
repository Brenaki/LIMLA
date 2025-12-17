"""
Definições dos modelos disponíveis para treinamento.
"""

from enum import Enum


class ModelType(str, Enum):
    """Enum com os tipos de modelos disponíveis."""
    MOBILENETV2 = "MobileNetV2"
    VGG16 = "VGG16"


# Lista de modelos suportados
SUPPORTED_MODELS = [ModelType.MOBILENETV2.value, ModelType.VGG16.value]
