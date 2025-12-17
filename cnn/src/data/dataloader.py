"""
DataLoaders e transformações de dados para treinamento.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple, List

from .dataset import ImageClassificationDataset


def get_transforms(is_training: bool = True) -> transforms.Compose:
    """
    Retorna transformações de dados para treinamento ou validação.
    
    Args:
        is_training: Se True, aplica data augmentation. Se False, apenas normalização.
        
    Returns:
        Compose com transformações
    """
    if is_training:
        # Transformações para treinamento com data augmentation
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]   # ImageNet std
            )
        ])
    else:
        # Transformações para validação/teste (sem augmentation)
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]   # ImageNet std
            )
        ])


def create_dataloaders(
    data_dir: str,
    quality: int,
    batch_size: int = 32,
    num_workers: int = 4,
    classes: Optional[list] = None
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Cria DataLoaders para treinamento e validação.
    
    Args:
        data_dir: Diretório base dos dados
        quality: Qualidade da imagem (1, 5, ou 10)
        batch_size: Tamanho do batch
        num_workers: Número de workers para carregamento de dados
        classes: Lista de classes (opcional, detecta automaticamente se None)
        
    Returns:
        Tupla (train_loader, val_loader, classes)
    """
    # Cria datasets
    train_dataset = ImageClassificationDataset(
        data_dir=data_dir,
        quality=quality,
        split='train',
        classes=classes,
        transform=get_transforms(is_training=True)
    )
    
    val_dataset = ImageClassificationDataset(
        data_dir=data_dir,
        quality=quality,
        split='val',
        classes=train_dataset.get_classes(),  # Usa as mesmas classes do treino
        transform=get_transforms(is_training=False)
    )
    
    # Obtém lista de classes
    classes = train_dataset.get_classes()
    
    # Cria DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, classes
