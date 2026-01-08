"""
Salvamento e carregamento de checkpoints do modelo.
"""

import os
import csv
from pathlib import Path
from datetime import datetime
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
    is_best: bool = False,
    history: Optional[Dict[str, List[float]]] = None
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
        history: Histórico de treinamento (opcional, para resume)
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
    
    # Adiciona histórico se fornecido
    if history is not None:
        checkpoint['history'] = history
    
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
    train_quality: int,
    test_quality: str,
    seed: Optional[int],
    history: Dict[str, List[float]],
    output_dir: str,
    split: str = 'train',
    n_train: Optional[int] = None,
    n_val: Optional[int] = None,
    n_test: Optional[int] = None,
    device: str = 'cpu'
) -> None:
    """
    Salva ou atualiza CSV com resultados do treinamento no formato longo.
    O CSV é salvo na pasta anterior ao diretório de saída.
    Formato longo: uma linha por combinação (split, metric, epoch_type).
    
    Args:
        model_name: Nome do modelo
        train_quality: Qualidade usada no treino (1-100)
        test_quality: Qualidade usada no teste (1-100 ou 'original')
        seed: Seed usado no treinamento (None se não fornecido)
        history: Dicionário com histórico de treinamento
        output_dir: Diretório de saída atual
        split: Split avaliado ('train', 'val', 'test')
        n_train: Número de amostras de treino
        n_val: Número de amostras de validação
        n_test: Número de amostras de teste
        device: Dispositivo usado ('cpu' ou 'cuda')
    """
    # Encontra a melhor época (menor val_loss)
    best_epoch_idx = min(
        range(len(history['val_loss'])),
        key=lambda i: history['val_loss'][i]
    )
    best_epoch = best_epoch_idx + 1  # Épocas começam em 1
    
    # Dados da última época
    last_epoch_idx = len(history['train_loss']) - 1
    last_epoch = last_epoch_idx + 1
    
    # Gera run_id
    seed_str = f"seed{seed}" if seed is not None else "noseed"
    run_id = f"{model_name}_q{train_quality}_{seed_str}"
    
    # Timestamp
    timestamp = datetime.now().isoformat()
    
    # Determina o caminho do CSV (pasta anterior ao output_dir)
    output_path = Path(output_dir)
    csv_dir = output_path.parent
    csv_path = csv_dir / 'tabela_resultados.csv'
    
    # Cabeçalho do CSV (formato longo)
    header = [
        'run_id', 'seed', 'model', 'train_quality', 'test_quality', 'split',
        'metric', 'value', 'epoch_type', 'epoch', 'timestamp',
        'n_train', 'n_val', 'n_test', 'device'
    ]
    
    # Verifica se o CSV já existe
    file_exists = csv_path.exists()
    
    # Prepara linhas de dados (formato longo: uma linha por métrica/época)
    rows = []
    
    # Para cada split disponível no history
    for current_split in ['train', 'val']:
        if current_split not in ['train', 'val']:
            continue
            
        # Métricas da melhor época
        if current_split == 'train':
            best_loss = history['train_loss'][best_epoch_idx]
            best_acc = history['train_accuracy'][best_epoch_idx]
        else:  # val
            best_loss = history['val_loss'][best_epoch_idx]
            best_acc = history['val_accuracy'][best_epoch_idx]
        
        # Métricas da última época
        if current_split == 'train':
            last_loss = history['train_loss'][last_epoch_idx]
            last_acc = history['train_accuracy'][last_epoch_idx]
        else:  # val
            last_loss = history['val_loss'][last_epoch_idx]
            last_acc = history['val_accuracy'][last_epoch_idx]
        
        # Loss - melhor época
        rows.append([
            run_id, seed_str, model_name, train_quality, test_quality, current_split,
            'loss', f'{best_loss:.6f}', 'best', best_epoch, timestamp,
            n_train, n_val, n_test, device
        ])
        
        # Accuracy - melhor época
        rows.append([
            run_id, seed_str, model_name, train_quality, test_quality, current_split,
            'accuracy', f'{best_acc:.6f}', 'best', best_epoch, timestamp,
            n_train, n_val, n_test, device
        ])
        
        # Loss - última época
        rows.append([
            run_id, seed_str, model_name, train_quality, test_quality, current_split,
            'loss', f'{last_loss:.6f}', 'last', last_epoch, timestamp,
            n_train, n_val, n_test, device
        ])
        
        # Accuracy - última época
        rows.append([
            run_id, seed_str, model_name, train_quality, test_quality, current_split,
            'accuracy', f'{last_acc:.6f}', 'last', last_epoch, timestamp,
            n_train, n_val, n_test, device
        ])
    
    # Abre o arquivo em modo append se existir, ou write se não existir
    mode = 'a' if file_exists else 'w'
    
    with open(csv_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Escreve cabeçalho apenas se for arquivo novo
        if not file_exists:
            writer.writerow(header)
        
        # Escreve todas as linhas de dados
        writer.writerows(rows)
    
    print(f"Resultados salvos em: {csv_path} (formato longo, {len(rows)} linhas adicionadas)")


def save_test_results_to_csv(
    model_name: str,
    train_quality: int,
    test_quality: str,
    seed: Optional[int],
    test_metrics: Dict[str, float],
    output_dir: str,
    best_epoch: int,
    n_train: Optional[int] = None,
    n_val: Optional[int] = None,
    n_test: Optional[int] = None,
    device: str = 'cpu'
) -> None:
    """
    Salva resultados do test split no CSV (formato longo).
    
    Args:
        model_name: Nome do modelo
        train_quality: Qualidade usada no treino
        test_quality: Qualidade usada no teste ('original' ou número)
        seed: Seed usado no treinamento
        test_metrics: Dicionário com métricas do test (ex: {'loss': 0.5, 'accuracy': 85.3})
        output_dir: Diretório de saída atual
        best_epoch: Época do melhor modelo
        n_train: Número de amostras de treino
        n_val: Número de amostras de validação
        n_test: Número de amostras de teste
        device: Dispositivo usado
    """
    # Gera run_id
    seed_str = f"seed{seed}" if seed is not None else "noseed"
    run_id = f"{model_name}_q{train_quality}_{seed_str}"
    
    # Timestamp
    timestamp = datetime.now().isoformat()
    
    # Determina o caminho do CSV
    output_path = Path(output_dir)
    csv_dir = output_path.parent
    csv_path = csv_dir / 'tabela_resultados.csv'
    
    # Cabeçalho do CSV
    header = [
        'run_id', 'seed', 'model', 'train_quality', 'test_quality', 'split',
        'metric', 'value', 'epoch_type', 'epoch', 'timestamp',
        'n_train', 'n_val', 'n_test', 'device'
    ]
    
    # Verifica se o CSV já existe
    file_exists = csv_path.exists()
    
    # Prepara linhas de dados
    rows = []
    for metric, value in test_metrics.items():
        rows.append([
            run_id, seed_str, model_name, train_quality, test_quality, 'test',
            metric, f'{value:.6f}', 'best', best_epoch, timestamp,
            n_train, n_val, n_test, device
        ])
    
    # Abre o arquivo em modo append
    mode = 'a' if file_exists else 'w'
    
    with open(csv_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Escreve cabeçalho apenas se for arquivo novo
        if not file_exists:
            writer.writerow(header)
        
        # Escreve todas as linhas de dados
        writer.writerows(rows)
    
    print(f"Resultados de teste salvos em: {csv_path} ({len(rows)} linhas adicionadas)")
