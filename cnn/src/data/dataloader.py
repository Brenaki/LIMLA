"""
DataLoaders e transformações de dados para treinamento.
"""

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple, List

from .dataset import ImageClassificationDataset


def worker_init_fn(worker_id: int) -> None:
    """
    Inicializa seed para cada worker do DataLoader.
    Garante reprodutibilidade mesmo com múltiplos workers.
    
    Args:
        worker_id: ID do worker
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
    classes: Optional[list] = None,
    seed: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Cria DataLoaders para treinamento e validação.
    
    Args:
        data_dir: Diretório base dos dados
        quality: Qualidade da imagem (1, 5, ou 10)
        batch_size: Tamanho do batch
        num_workers: Número de workers para carregamento de dados
        classes: Lista de classes (opcional, detecta automaticamente se None)
        seed: Seed para reprodutibilidade (opcional)
        
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
    
    # Configura generator para reprodutibilidade se seed fornecido
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    
    # Cria DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        worker_init_fn=worker_init_fn if seed is not None else None,
        generator=generator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        worker_init_fn=worker_init_fn if seed is not None else None,
        generator=generator
    )
    
    return train_loader, val_loader, classes


def create_test_loader(
    data_dir: str,
    test_quality: str,
    batch_size: int = 32,
    num_workers: int = 4,
    classes: Optional[list] = None,
    seed: Optional[int] = None,
    original_data_dir: Optional[str] = None
) -> Tuple[DataLoader, List[str]]:
    """
    Cria DataLoader para o conjunto de teste.
    Suporta test_quality diferente de train_quality, incluindo 'original'.
    
    Args:
        data_dir: Diretório base dos dados comprimidos
        test_quality: Qualidade da imagem para teste ('original' ou número 1-100)
        batch_size: Tamanho do batch
        num_workers: Número de workers para carregamento de dados
        classes: Lista de classes (opcional, detecta automaticamente se None)
        seed: Seed para reprodutibilidade (opcional)
        original_data_dir: Diretório com imagens originais (sem compressão) se test_quality='original'
        
    Returns:
        Tupla (test_loader, classes)
    """
    # Se test_quality é 'original', usa original_data_dir ou estrutura especial
    if test_quality == 'original':
        if original_data_dir:
            # Usa diretório original fornecido
            # Assume estrutura: original_data_dir/test/{class}/
            custom_path = str(Path(original_data_dir) / 'test')
            test_dataset = ImageClassificationDataset(
                data_dir=original_data_dir,  # Base dir
                quality=100,  # Dummy, não usado
                split='test',
                classes=classes,
                transform=get_transforms(is_training=False),
                custom_split_path=custom_path
            )
        else:
            # Tenta encontrar em compressed/original/test/ ou data_dir/original/test/
            possible_paths = [
                Path(data_dir).parent / 'original' / 'test',
                Path(data_dir) / 'original' / 'test'
            ]
            
            custom_path = None
            for path in possible_paths:
                if path.exists():
                    custom_path = str(path)
                    break
            
            if custom_path:
                test_dataset = ImageClassificationDataset(
                    data_dir=str(Path(custom_path).parent.parent),  # Base dir
                    quality=100,  # Dummy
                    split='test',
                    classes=classes,
                    transform=get_transforms(is_training=False),
                    custom_split_path=custom_path
                )
            else:
                # Fallback: usa qualidade máxima disponível como proxy
                print(f"AVISO: Diretório original não encontrado. Tentando caminhos: {possible_paths}")
                print("Usando qualidade 100 como proxy para original")
                test_dataset = ImageClassificationDataset(
                    data_dir=data_dir,
                    quality=100,
                    split='test',
                    classes=classes,
                    transform=get_transforms(is_training=False)
                )
    else:
        # Qualidade numérica
        quality_int = int(test_quality)
        test_dataset = ImageClassificationDataset(
            data_dir=data_dir,
            quality=quality_int,
            split='test',
            classes=classes,
            transform=get_transforms(is_training=False)
        )
    
    # Obtém lista de classes
    classes = test_dataset.get_classes()
    
    # Configura generator para reprodutibilidade se seed fornecido
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    
    # Cria DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        worker_init_fn=worker_init_fn if seed is not None else None,
        generator=generator
    )
    
    return test_loader, classes
