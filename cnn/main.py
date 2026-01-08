"""
Ponto de entrada principal para treinamento de modelos CNN.
Orquestra todas as camadas seguindo Clean Architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.presentation.cli import parse_args
from src.presentation.progress import print_config, print_training_summary
from src.data.dataloader import create_dataloaders
from src.infrastructure.model_builder import build_model
from src.infrastructure.checkpoint import save_checkpoint, save_classes_mapping, save_results_to_csv
from src.use_cases.train import train_model
from src.use_cases.early_stopping import EarlyStopping


def main():
    """Função principal."""
    # Parse argumentos
    args = parse_args()
    
    # Exibe configuração
    print_config(args)
    
    # Configura dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}\n")
    
    # Cria DataLoaders
    print("Carregando dados...")
    train_loader, val_loader, classes = create_dataloaders(
        data_dir=args.data_dir,
        quality=args.quality,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    num_classes = len(classes)
    print(f"Classes detectadas ({num_classes}): {', '.join(classes)}")
    print(f"Imagens de treinamento: {len(train_loader.dataset)}")
    print(f"Imagens de validação: {len(val_loader.dataset)}\n")
    
    # Valida número de classes se fornecido
    if args.num_classes and args.num_classes != num_classes:
        print(
            f"AVISO: Número de classes fornecido ({args.num_classes}) "
            f"diferente do detectado ({num_classes}). Usando {num_classes}."
        )
    
    # Constrói modelo
    print(f"Construindo modelo {args.model}...")
    model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=True
    )
    model = model.to(device)
    
    # Conta parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total de parâmetros: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}\n")
    
    # Configura otimizador e loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Configura early stopping
    output_dir = Path(args.output_dir)
    model_output_dir = output_dir / args.model
    best_model_path = model_output_dir / 'best.pt'
    
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        mode='min',  # Minimiza val_loss
        verbose=True,
        save_path=str(best_model_path)
    )
    
    # Treina modelo
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        early_stopping=early_stopping
    )
    
    # Salva melhor modelo e último checkpoint
    print("Salvando modelos...")
    best_epoch_idx = min(
        range(len(history['val_loss'])),
        key=lambda i: history['val_loss'][i]
    )
    
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=best_epoch_idx,
        loss=history['val_loss'][best_epoch_idx],
        accuracy=history['val_accuracy'][best_epoch_idx],
        output_dir=str(output_dir),
        model_name=args.model,
        is_best=True
    )
    
    # Salva último checkpoint
    last_epoch = len(history['train_loss']) - 1
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=last_epoch,
        loss=history['val_loss'][last_epoch],
        accuracy=history['val_accuracy'][last_epoch],
        output_dir=str(output_dir),
        model_name=args.model,
        is_best=False
    )
    
    # Salva mapeamento de classes
    save_classes_mapping(
        classes=classes,
        output_dir=str(output_dir),
        model_name=args.model
    )
    
    print(f"Modelos salvos em: {model_output_dir}")
    
    # Salva resultados no CSV consolidado
    save_results_to_csv(
        model_name=args.model,
        quality=args.quality,
        history=history,
        output_dir=str(output_dir)
    )
    
    # Exibe resumo
    print_training_summary(history)
    
    print("Treinamento concluído!")


if __name__ == '__main__':
    main()
