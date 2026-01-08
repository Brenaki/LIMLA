"""
Salvamento e carregamento de checkpoints do modelo.
"""

import os
import csv
from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional, Dict, List


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


def save_results_to_csv(
    model_name: str,
    quality: int,
    history: Dict[str, List[float]],
    output_dir: str
) -> None:
    """
    Salva ou atualiza CSV com resultados do treinamento.
    O CSV é salvo na pasta anterior ao diretório de saída.
    Se o CSV já existir, adiciona uma nova linha.
    
    Args:
        model_name: Nome do modelo
        quality: Qualidade da imagem (1-100)
        history: Dicionário com histórico de treinamento
        output_dir: Diretório de saída atual
    """
    # Encontra a melhor época (menor val_loss)
    best_epoch_idx = min(
        range(len(history['val_loss'])),
        key=lambda i: history['val_loss'][i]
    )
    
    # Dados da melhor época
    best_epoch = best_epoch_idx + 1  # Épocas começam em 1
    best_train_loss = history['train_loss'][best_epoch_idx]
    best_train_acc = history['train_accuracy'][best_epoch_idx]
    best_val_loss = history['val_loss'][best_epoch_idx]
    best_val_acc = history['val_accuracy'][best_epoch_idx]
    
    # Dados da última época
    last_epoch_idx = len(history['train_loss']) - 1
    last_epoch = last_epoch_idx + 1
    last_train_loss = history['train_loss'][last_epoch_idx]
    last_train_acc = history['train_accuracy'][last_epoch_idx]
    last_val_loss = history['val_loss'][last_epoch_idx]
    last_val_acc = history['val_accuracy'][last_epoch_idx]
    
    # Determina o caminho do CSV (pasta anterior ao output_dir)
    output_path = Path(output_dir)
    csv_dir = output_path.parent
    csv_path = csv_dir / 'tabela_resultados.csv'
    
    # Prepara a linha de dados
    row = [
        model_name,
        f'q{quality}',
        'Melhor Época',
        str(best_epoch),
        f'{best_train_loss:.4f}',
        f'{best_train_acc:.2f}%',
        f'{best_val_loss:.4f}',
        f'{best_val_acc:.2f}%',
        str(last_epoch),
        f'{last_train_loss:.4f}',
        f'{last_train_acc:.2f}%',
        f'{last_val_loss:.4f}',
        f'{last_val_acc:.2f}%'
    ]
    
    # Cabeçalho do CSV
    header = [
        'Modelo', 'Qualidade', 'Métrica', 'Melhor Época',
        'Train Loss (Melhor)', 'Train Accuracy (Melhor)',
        'Val Loss (Melhor)', 'Val Accuracy (Melhor)',
        'Última Época', 'Train Loss (Última)', 'Train Accuracy (Última)',
        'Val Loss (Última)', 'Val Accuracy (Última)'
    ]
    
    # Verifica se o CSV já existe
    file_exists = csv_path.exists()
    
    # Abre o arquivo em modo append se existir, ou write se não existir
    mode = 'a' if file_exists else 'w'
    
    with open(csv_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Escreve cabeçalho apenas se for arquivo novo
        if not file_exists:
            writer.writerow(header)
        
        # Escreve a linha de dados
        writer.writerow(row)
    
    print(f"Resultados salvos em: {csv_path}")
