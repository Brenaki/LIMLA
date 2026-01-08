"""
Ponto de entrada principal para treinamento de modelos CNN.
Orquestra todas as camadas seguindo Clean Architecture.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.presentation.cli import parse_args
from src.presentation.progress import print_config, print_training_summary
from src.data.dataloader import create_dataloaders, create_test_loader
from src.infrastructure.model_builder import build_model
from src.infrastructure.checkpoint import (
    save_checkpoint, save_classes_mapping, save_results_to_csv,
    save_test_results_to_csv, load_checkpoint
)
from src.use_cases.train import train_model, evaluate_test
from src.use_cases.early_stopping import EarlyStopping


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Configura seeds para reprodutibilidade.
    
    Args:
        seed: Valor do seed
        deterministic: Se True, habilita modo totalmente determinístico (mais lento)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Cuidado: use_deterministic_algorithms pode quebrar algumas operações
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(f"AVISO: Não foi possível habilitar algoritmos determinísticos: {e}")


def main():
    """Função principal."""
    # Parse argumentos
    args = parse_args()
    
    # Configura seed se fornecido
    if args.seed is not None:
        set_seed(args.seed, args.deterministic)
        print(f"Seed configurado: {args.seed} (determinístico: {args.deterministic})")
    
    # Exibe configuração
    print_config(args)
    
    # Configura dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}\n")
    
    # Cria DataLoaders
    print("Carregando dados...")
    train_loader, val_loader, classes = create_dataloaders(
        data_dir=args.data_dir,
        quality=args.quality,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    num_classes = len(classes)
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    print(f"Classes detectadas ({num_classes}): {', '.join(classes)}")
    print(f"Imagens de treinamento: {n_train}")
    print(f"Imagens de validação: {n_val}\n")
    
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
    # Inclui seed no nome do diretório do modelo se fornecido
    if args.seed is not None:
        model_dir_name = f"{args.model}_seed{args.seed}"
    else:
        model_dir_name = args.model
    model_output_dir = output_dir / model_dir_name
    best_model_path = model_output_dir / 'best.pt'
    last_checkpoint_path = model_output_dir / 'last.pt'
    
    # Verifica se deve continuar de um checkpoint
    start_epoch = 0
    history = None
    resume_from = None
    
    if args.resume:
        # Resume do checkpoint especificado
        resume_from = Path(args.resume)
        if not resume_from.exists():
            print(f"ERRO: Checkpoint não encontrado: {resume_from}")
            return
    elif last_checkpoint_path.exists():
        # Detecta automaticamente checkpoint existente
        resume_from = last_checkpoint_path
        print(f"Checkpoint encontrado: {resume_from}")
        print("Continuando treinamento do checkpoint...")
    
    # Carrega checkpoint se encontrado
    if resume_from:
        print(f"Carregando checkpoint de: {resume_from}")
        checkpoint = load_checkpoint(str(resume_from), model, optimizer)
        start_epoch = checkpoint.get('epoch', 0) + 1  # Próxima época a treinar
        history = checkpoint.get('history', None)
        # Garante que modelo está no device correto
        model = model.to(device)
        print(f"Checkpoint carregado: época {checkpoint.get('epoch', 0) + 1}, "
              f"loss={checkpoint.get('loss', 0):.4f}, "
              f"accuracy={checkpoint.get('accuracy', 0):.2f}%")
        print(f"Continuando da época {start_epoch + 1}/{args.epochs}")
    
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        mode='min',  # Minimiza val_loss
        verbose=True,
        save_path=str(best_model_path)
    )
    
    # Callback para salvar checkpoint periódico
    def save_periodic_checkpoint(epoch, model, optimizer, history):
        """Salva checkpoint a cada N épocas."""
        if (epoch + 1) % args.checkpoint_interval == 0:
            val_loss = history['val_loss'][-1] if history['val_loss'] else 0.0
            val_acc = history['val_accuracy'][-1] if history['val_accuracy'] else 0.0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                accuracy=val_acc,
                output_dir=str(output_dir),
                model_name=model_dir_name,
                is_best=False,
                history=history
            )
            print(f"  Checkpoint salvo (época {epoch + 1})")
    
    # Treina modelo
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        early_stopping=early_stopping,
        start_epoch=start_epoch,
        history=history,
        checkpoint_callback=save_periodic_checkpoint
    )
    
    # Salva melhor modelo e último checkpoint
    print("Salvando modelos...")
    if len(history['val_loss']) > 0:
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
            model_name=model_dir_name,
            is_best=True,
            history=history
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
            model_name=model_dir_name,
            is_best=False,
            history=history
        )
    else:
        print("AVISO: Nenhum histórico disponível para salvar.")
    
    # Salva mapeamento de classes
    save_classes_mapping(
        classes=classes,
        output_dir=str(output_dir),
        model_name=model_dir_name
    )
    
    print(f"Modelos salvos em: {model_output_dir}")
    
    # Salva resultados no CSV consolidado (formato longo)
    save_results_to_csv(
        model_name=args.model,
        train_quality=args.quality,
        test_quality=str(args.quality),  # Por padrão, test_quality = train_quality
        seed=args.seed,
        history=history,
        output_dir=str(output_dir),
        split='train',  # Será sobrescrito internamente para train/val
        n_train=n_train,
        n_val=n_val,
        n_test=None,  # Será preenchido quando avaliar test split
        device=device_str
    )
    
    # Exibe resumo
    print_training_summary(history)
    
    # Avalia no test split com múltiplas qualidades
    print("\n" + "="*60)
    print("AVALIAÇÃO NO TEST SPLIT")
    print("="*60)
    
    # Carrega melhor modelo
    best_model_path = model_output_dir / 'best.pt'
    if best_model_path.exists():
        print(f"\nCarregando melhor modelo de: {best_model_path}")
        checkpoint = load_checkpoint(str(best_model_path), model, optimizer=None)
        best_epoch = checkpoint.get('epoch', best_epoch_idx) + 1
        # Garante que modelo está no device correto
        model = model.to(device)
    else:
        print("AVISO: Melhor modelo não encontrado, usando modelo atual")
        best_epoch = best_epoch_idx + 1
    
    # Detecta automaticamente todas as qualidades disponíveis para teste
    data_dir_path = Path(args.data_dir)
    test_qualities = []
    
    # Detecta qualidades numéricas (q1, q5, q10, etc)
    available_qualities = []
    for item in data_dir_path.iterdir():
        if item.is_dir() and item.name.startswith('q') and item.name[1:].isdigit():
            quality_num = int(item.name[1:])
            # Verifica se tem pasta test
            test_dir = item / 'test'
            if test_dir.exists():
                available_qualities.append(quality_num)
    
    # Ordena as qualidades
    available_qualities.sort()
    
    # Adiciona todas as qualidades disponíveis
    for q in available_qualities:
        test_qualities.append(str(q))
    
    # Detecta pasta original (qualquer pasta que não seja q* e tenha test/)
    original_dir = None
    for item in data_dir_path.iterdir():
        if item.is_dir() and not item.name.startswith('q'):
            test_dir = item / 'test'
            if test_dir.exists():
                original_dir = item
                break
    
    # Se encontrou pasta original, adiciona 'original' à lista
    if original_dir:
        test_qualities.append('original')
        print(f"Pasta original detectada: {original_dir.name}")
    
    print(f"Qualidades disponíveis para teste: {test_qualities}")
    
    criterion = nn.CrossEntropyLoss()
    
    for test_q in test_qualities:
        print(f"\nAvaliando com test_quality = {test_q}...")
        try:
            # Cria test loader
            # Se for 'original', passa o diretório original detectado
            original_data_dir = None
            if test_q == 'original' and original_dir:
                original_data_dir = str(original_dir)
            
            test_loader, test_classes = create_test_loader(
                data_dir=args.data_dir,
                test_quality=test_q,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                classes=classes,  # Usa classes do treino
                seed=args.seed,
                original_data_dir=original_data_dir
            )
            
            n_test = len(test_loader.dataset)
            print(f"Imagens de teste: {n_test}")
            
            # Avalia
            test_metrics = evaluate_test(
                model=model,
                dataloader=test_loader,
                criterion=criterion,
                device=device
            )
            
            print(f"  Test ({test_q}) - Loss: {test_metrics['loss']:.4f}, "
                  f"Accuracy: {test_metrics['accuracy']:.2f}%")
            
            # Salva resultados no CSV
            save_test_results_to_csv(
                model_name=args.model,
                train_quality=args.quality,
                test_quality=test_q,
                seed=args.seed,
                test_metrics=test_metrics,
                output_dir=str(output_dir),
                best_epoch=best_epoch,
                n_train=n_train,
                n_val=n_val,
                n_test=n_test,
                device=device_str
            )
            
        except Exception as e:
            print(f"ERRO ao avaliar com test_quality={test_q}: {e}")
            print("Continuando com próxima qualidade...")
            continue
    
    print("\n" + "="*60)
    print("Treinamento e avaliação concluídos!")


if __name__ == '__main__':
    main()
