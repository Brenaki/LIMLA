"""
Dataset PyTorch customizado para classificação de imagens.
Carrega imagens da estrutura: {data_dir}/q{quality}/{split}/{class}/*.jpg
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset


class ImageClassificationDataset(Dataset):
    """
    Dataset para classificação de imagens.
    
    Carrega imagens de uma estrutura de pastas organizada por classes.
    Estrutura esperada: {data_dir}/q{quality}/{split}/{class_name}/*.jpg
    """
    
    def __init__(
        self,
        data_dir: str,
        quality: int,
        split: str,
        classes: Optional[List[str]] = None,
        transform: Optional[callable] = None
    ):
        """
        Inicializa o dataset.
        
        Args:
            data_dir: Diretório base dos dados (ex: ./compressed)
            quality: Qualidade da imagem (1, 5, ou 10)
            split: Split a carregar (train, val, test)
            classes: Lista de classes. Se None, detecta automaticamente
            transform: Transformações a aplicar nas imagens
        """
        self.data_dir = Path(data_dir)
        self.quality = quality
        self.split = split
        self.transform = transform
        
        # Constrói caminho: {data_dir}/q{quality}/{split}/
        quality_dir = self.data_dir / f"q{quality}" / split
        
        if not quality_dir.exists():
            raise ValueError(
                f"Diretório não encontrado: {quality_dir}. "
                f"Verifique se --data_dir e --quality estão corretos."
            )
        
        # Detecta classes se não fornecidas
        if classes is None:
            self.classes = sorted([
                d.name for d in quality_dir.iterdir() 
                if d.is_dir()
            ])
        else:
            self.classes = classes
        
        if not self.classes:
            raise ValueError(
                f"Nenhuma classe encontrada em {quality_dir}. "
                f"Verifique a estrutura de pastas."
            )
        
        # Cria mapeamento classe -> índice
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Carrega lista de (caminho_imagem, índice_classe)
        self.samples = self._load_samples(quality_dir)
        
        if not self.samples:
            raise ValueError(
                f"Nenhuma imagem encontrada em {quality_dir}. "
                f"Verifique se há arquivos .jpg nas pastas das classes."
            )
    
    def _load_samples(self, quality_dir: Path) -> List[Tuple[str, int]]:
        """
        Carrega lista de (caminho_imagem, índice_classe).
        
        Args:
            quality_dir: Diretório do split (ex: compressed/q1/train/)
            
        Returns:
            Lista de tuplas (caminho_imagem, índice_classe)
        """
        samples = []
        extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
        for class_name in self.classes:
            class_dir = quality_dir / class_name
            if not class_dir.exists():
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Busca todas as imagens na pasta da classe
            for img_path in class_dir.iterdir():
                if img_path.suffix in extensions:
                    samples.append((str(img_path), class_idx))
        
        return samples
    
    def __len__(self) -> int:
        """Retorna o tamanho do dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retorna uma amostra do dataset.
        
        Args:
            idx: Índice da amostra
            
        Returns:
            Tupla (imagem_tensor, classe_idx)
        """
        img_path, class_idx = self.samples[idx]
        
        # Carrega imagem
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar imagem {img_path}: {e}")
        
        # Aplica transformações
        if self.transform:
            image = self.transform(image)
        
        return image, class_idx
    
    def get_classes(self) -> List[str]:
        """Retorna lista de classes."""
        return self.classes
