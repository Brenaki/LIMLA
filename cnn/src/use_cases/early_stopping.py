"""
Early stopping customizado para parar treinamento quando não há melhoria.
"""

from typing import Optional
import os
from pathlib import Path
import torch
import torch.nn as nn


class EarlyStopping:
    """
    Early stopping que monitora uma métrica e para o treinamento
    quando não há melhoria por um número de épocas (patience).
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = 'min',
        verbose: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Inicializa early stopping.
        
        Args:
            patience: Número de épocas sem melhoria antes de parar
            min_delta: Melhoria mínima considerada como progresso
            mode: 'min' para minimizar (loss) ou 'max' para maximizar (accuracy)
            verbose: Se True, imprime mensagens
            save_path: Caminho para salvar melhor modelo (opcional)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.save_path = save_path
        
        self.best_value: Optional[float] = None
        self.counter = 0
        self.best_epoch = 0
        self.best_model_state: Optional[dict] = None
        self.stopped_epoch = 0
        
        # Determina função de comparação
        if mode == 'min':
            self.is_better = lambda current, best: current < (best - min_delta)
        elif mode == 'max':
            self.is_better = lambda current, best: current > (best + min_delta)
        else:
            raise ValueError(f"Mode deve ser 'min' ou 'max', recebido: {mode}")
    
    def __call__(self, epoch: int, value: float, model: nn.Module) -> bool:
        """
        Verifica se deve parar o treinamento.
        
        Args:
            epoch: Época atual
            value: Valor da métrica monitorada
            model: Modelo atual
            
        Returns:
            True se deve parar, False caso contrário
        """
        # Inicializa melhor valor na primeira chamada
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
            self.best_model_state = model.state_dict().copy()
            if self.save_path:
                self._save_model(model, epoch, value)
            return False
        
        # Verifica se há melhoria
        if self.is_better(value, self.best_value):
            # Melhoria detectada
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
            
            if self.save_path:
                self._save_model(model, epoch, value)
            
            if self.verbose:
                print(f"  Melhoria detectada! Melhor {self.mode}: {value:.4f} (época {epoch + 1})")
        else:
            # Sem melhoria
            self.counter += 1
            if self.verbose:
                print(
                    f"  Sem melhoria ({self.counter}/{self.patience}). "
                    f"Melhor: {self.best_value:.4f} na época {self.best_epoch + 1}"
                )
        
        # Verifica se deve parar
        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            if self.verbose:
                print(
                    f"\nEarly stopping ativado na época {epoch + 1}. "
                    f"Melhor época: {self.best_epoch + 1} com {self.mode} = {self.best_value:.4f}"
                )
            return True
        
        return False
    
    def restore_best_model(self, model: nn.Module) -> None:
        """
        Restaura os pesos do melhor modelo.
        
        Args:
            model: Modelo onde restaurar os pesos
        """
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"Pesos restaurados da melhor época ({self.best_epoch + 1})")
    
    def _save_model(self, model: nn.Module, epoch: int, value: float) -> None:
        """Salva modelo em disco."""
        if self.save_path:
            path = Path(self.save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'value': value
            }, path)
