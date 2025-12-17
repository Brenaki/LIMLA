"""
Interfaces/contratos abstratos seguindo Clean Architecture.
Define os contratos que as camadas de infraestrutura devem implementar.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import torch
import torch.nn as nn


class IModelBuilder(ABC):
    """Interface para construção de modelos."""
    
    @abstractmethod
    def build(self, model_name: str, num_classes: int) -> nn.Module:
        """
        Constrói um modelo de classificação de imagens.
        
        Args:
            model_name: Nome do modelo (MobileNetV2 ou VGG16)
            num_classes: Número de classes para classificação
            
        Returns:
            Modelo PyTorch configurado
        """
        pass


class IDatasetLoader(ABC):
    """Interface para carregamento de datasets."""
    
    @abstractmethod
    def load_dataset(self, data_dir: str, quality: int, split: str) -> List[Tuple[str, int]]:
        """
        Carrega lista de (caminho_imagem, classe) para um split específico.
        
        Args:
            data_dir: Diretório base dos dados
            quality: Qualidade da imagem (1, 5, ou 10)
            split: Split a carregar (train, val, test)
            
        Returns:
            Lista de tuplas (caminho_imagem, índice_classe)
        """
        pass
    
    @abstractmethod
    def get_classes(self, data_dir: str, quality: int) -> List[str]:
        """
        Obtém lista de classes a partir da estrutura de pastas.
        
        Args:
            data_dir: Diretório base dos dados
            quality: Qualidade da imagem (1, 5, ou 10)
            
        Returns:
            Lista ordenada de nomes das classes
        """
        pass


class ITrainer(ABC):
    """Interface para treinamento de modelos."""
    
    @abstractmethod
    def train_epoch(self, model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                   criterion: nn.Module, optimizer: torch.optim.Optimizer, 
                   device: torch.device) -> Dict[str, float]:
        """
        Executa uma época de treinamento.
        
        Args:
            model: Modelo a ser treinado
            dataloader: DataLoader com dados de treinamento
            criterion: Função de perda
            optimizer: Otimizador
            device: Dispositivo (CPU ou GPU)
            
        Returns:
            Dicionário com métricas da época (loss, accuracy)
        """
        pass
    
    @abstractmethod
    def validate_epoch(self, model: nn.Module, dataloader: torch.utils.data.DataLoader,
                      criterion: nn.Module, device: torch.device) -> Dict[str, float]:
        """
        Executa validação em uma época.
        
        Args:
            model: Modelo a ser validado
            dataloader: DataLoader com dados de validação
            criterion: Função de perda
            device: Dispositivo (CPU ou GPU)
            
        Returns:
            Dicionário com métricas da validação (loss, accuracy)
        """
        pass
