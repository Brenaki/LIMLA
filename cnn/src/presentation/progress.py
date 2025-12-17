"""
Exibição de progresso e métricas durante o treinamento.
"""

from typing import Dict, List


def print_training_summary(history: Dict[str, List[float]]) -> None:
    """
    Imprime resumo do treinamento.
    
    Args:
        history: Dicionário com histórico de treinamento
    """
    if not history['train_loss']:
        return
    
    print("\n" + "="*60)
    print("RESUMO DO TREINAMENTO")
    print("="*60)
    
    # Melhor época (menor val_loss)
    best_epoch_idx = min(
        range(len(history['val_loss'])),
        key=lambda i: history['val_loss'][i]
    )
    
    print(f"\nMelhor época: {best_epoch_idx + 1}")
    print(f"  Train - Loss: {history['train_loss'][best_epoch_idx]:.4f}, "
          f"Accuracy: {history['train_accuracy'][best_epoch_idx]:.2f}%")
    print(f"  Val   - Loss: {history['val_loss'][best_epoch_idx]:.4f}, "
          f"Accuracy: {history['val_accuracy'][best_epoch_idx]:.2f}%")
    
    # Última época
    last_epoch = len(history['train_loss']) - 1
    print(f"\nÚltima época: {last_epoch + 1}")
    print(f"  Train - Loss: {history['train_loss'][last_epoch]:.4f}, "
          f"Accuracy: {history['train_accuracy'][last_epoch]:.2f}%")
    print(f"  Val   - Loss: {history['val_loss'][last_epoch]:.4f}, "
          f"Accuracy: {history['val_accuracy'][last_epoch]:.2f}%")
    
    print("="*60 + "\n")


def print_config(args) -> None:
    """
    Imprime configuração do treinamento.
    
    Args:
        args: Namespace com argumentos
    """
    print("="*60)
    print("CONFIGURAÇÃO DO TREINAMENTO")
    print("="*60)
    print(f"Modelo: {args.model}")
    print(f"Diretório de dados: {args.data_dir}")
    print(f"Qualidade: q{args.quality}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Early stopping - Patience: {args.patience}, Min delta: {args.min_delta}")
    print(f"Diretório de saída: {args.output_dir}")
    print("="*60 + "\n")
