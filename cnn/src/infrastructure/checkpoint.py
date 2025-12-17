"""
Salvamento e carregamento de checkpoints do modelo.
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional, Dict


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    output_dir: str,
    model_name: str,
    is_best: bool = False
) -> None:
    """
    Salva checkpoint do modelo.
    
    Args:
        model: Modelo a ser salvo
        optimizer: Otimizador
        epoch: Época atual
        loss: Loss atual
        accuracy: Accuracy atual
        output_dir: Diretório de saída
        model_name: Nome do modelo (para organização)
        is_best: Se True, salva como melhor modelo
    """
    # Cria diretório de saída
    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepara dados do checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    
    # Salva último checkpoint
    last_path = model_dir / 'last.pt'
    torch.save(checkpoint, last_path)
    
    # Salva melhor modelo se indicado
    if is_best:
        best_path = model_dir / 'best.pt'
        torch.save(checkpoint, best_path)
        
        # Salva também informações adicionais
        info_path = model_dir / 'info.json'
        import json
        with open(info_path, 'w') as f:
            json.dump({
                'model_name': model_name,
                'best_epoch': epoch,
                'best_loss': loss,
                'best_accuracy': accuracy
            }, f, indent=2)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict:
    """
    Carrega checkpoint do modelo.
    
    Args:
        checkpoint_path: Caminho do checkpoint
        model: Modelo onde carregar os pesos
        optimizer: Otimizador (opcional)
        
    Returns:
        Dicionário com informações do checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Carrega pesos do modelo
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Carrega estado do optimizer se fornecido
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def save_classes_mapping(classes: list, output_dir: str, model_name: str) -> None:
    """
    Salva mapeamento de classes em arquivo JSON.
    
    Args:
        classes: Lista de classes
        output_dir: Diretório de saída
        model_name: Nome do modelo
    """
    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    classes_path = model_dir / 'classes.json'
    
    # Cria mapeamento índice -> classe
    classes_dict = {idx: cls for idx, cls in enumerate(classes)}
    
    with open(classes_path, 'w') as f:
        json.dump(classes_dict, f, indent=2)
