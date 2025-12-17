"""
Lógica de treinamento e validação de modelos.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
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
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Barra de progresso
    pbar = tqdm(dataloader, desc="Treinamento", leave=False)
    
    for images, labels in pbar:
        # Move dados para dispositivo
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calcula métricas
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Atualiza barra de progresso
        current_loss = running_loss / len(dataloader)
        current_acc = 100 * correct / total
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100 * correct / total
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_accuracy
    }


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
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
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Barra de progresso
    pbar = tqdm(dataloader, desc="Validação", leave=False)
    
    with torch.no_grad():
        for images, labels in pbar:
            # Move dados para dispositivo
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calcula métricas
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Atualiza barra de progresso
            current_loss = running_loss / len(dataloader)
            current_acc = 100 * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100 * correct / total
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_accuracy
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    early_stopping = None
) -> Dict[str, List[float]]:
    """
    Loop principal de treinamento.
    
    Args:
        model: Modelo a ser treinado
        train_loader: DataLoader de treinamento
        val_loader: DataLoader de validação
        criterion: Função de perda
        optimizer: Otimizador
        device: Dispositivo (CPU ou GPU)
        epochs: Número máximo de épocas
        early_stopping: Instância de EarlyStopping (opcional)
        
    Returns:
        Dicionário com histórico de treinamento
    """
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    print(f"\nIniciando treinamento por {epochs} épocas...")
    print(f"Dispositivo: {device}\n")
    
    for epoch in range(epochs):
        print(f"Época {epoch + 1}/{epochs}")
        
        # Treinamento
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        
        # Validação
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        
        # Exibe resumo da época
        print(
            f"  Train - Loss: {train_metrics['loss']:.4f}, "
            f"Accuracy: {train_metrics['accuracy']:.2f}%"
        )
        print(
            f"  Val   - Loss: {val_metrics['loss']:.4f}, "
            f"Accuracy: {val_metrics['accuracy']:.2f}%"
        )
        
        # Verifica early stopping
        if early_stopping:
            should_stop = early_stopping(
                epoch=epoch,
                value=val_metrics['loss'],  # Monitora val_loss
                model=model
            )
            if should_stop:
                print("\nTreinamento interrompido por early stopping.")
                early_stopping.restore_best_model(model)
                break
        
        print()
    
    return history
