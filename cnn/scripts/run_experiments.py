"""
Script para executar grid de experimentos de forma idempotente.
Verifica CSV antes de rodar para evitar retrabalho.
"""

import subprocess
import sys
import pandas as pd
from pathlib import Path
from typing import List, Optional
import logging
from datetime import datetime


def setup_logging(log_dir: Path, model: str, train_quality: int, seed: int) -> logging.Logger:
    """
    Configura logging para uma run específica.
    
    Args:
        log_dir: Diretório para logs
        model: Nome do modelo
        train_quality: Qualidade de treino
        seed: Seed usado
        
    Returns:
        Logger configurado
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{model}_q{train_quality}_seed{seed}.log"
    
    logger = logging.getLogger(f"{model}_q{train_quality}_seed{seed}")
    logger.setLevel(logging.INFO)
    
    # Handler para arquivo
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def check_if_already_run(
    csv_path: Path,
    model: str,
    train_quality: int,
    seed: int
) -> bool:
    """
    Verifica se uma combinação já foi executada.
    
    Args:
        csv_path: Caminho para o CSV
        model: Nome do modelo
        train_quality: Qualidade de treino
        seed: Seed usado
        
    Returns:
        True se já existe, False caso contrário
    """
    if not csv_path.exists():
        return False
    
    try:
        df = pd.read_csv(csv_path)
        
        # Verifica se existe linha com esta combinação
        # Para test split, best epoch, accuracy
        run_id = f"{model}_q{train_quality}_seed{seed}"
        
        exists = (
            (df['run_id'] == run_id) &
            (df['split'] == 'test') &
            (df['epoch_type'] == 'best') &
            (df['metric'] == 'accuracy')
        ).any()
        
        return exists
    except Exception as e:
        print(f"ERRO ao verificar CSV: {e}")
        return False


def run_training(
    model: str,
    train_quality: int,
    seed: int,
    data_dir: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 5,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Executa treinamento de um modelo.
    
    Args:
        model: Nome do modelo
        train_quality: Qualidade de treino
        seed: Seed usado
        data_dir: Diretório dos dados
        output_dir: Diretório de saída
        epochs: Número de épocas
        batch_size: Tamanho do batch
        patience: Paciência para early stopping
        logger: Logger (opcional)
        
    Returns:
        True se sucesso, False caso contrário
    """
    log = logger.info if logger else print
    
    log(f"Iniciando treinamento: {model}, q={train_quality}, seed={seed}")
    
    cmd = [
        sys.executable,
        'main.py',
        '--model', model,
        '--data_dir', data_dir,
        '--quality', str(train_quality),
        '--seed', str(seed),
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--patience', str(patience),
        '--output_dir', output_dir
    ]
    
    try:
        # Executa processo e exibe saída em tempo real
        # Usa Popen para ler linha por linha e exibir imediatamente
        process = subprocess.Popen(
            cmd,
            cwd=Path(__file__).parent.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Lê e exibe saída em tempo real
        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            if line:
                print(line, flush=True)  # Exibe imediatamente
                output_lines.append(line)
                if logger:
                    logger.info(line)
        
        # Espera processo terminar
        return_code = process.wait()
        
        if return_code != 0:
            error_msg = f"ERRO no treinamento: código de saída {return_code}"
            if logger:
                logger.error(error_msg)
            else:
                print(error_msg)
            return False
        
        log(f"Treinamento concluído com sucesso")
        return True
        
    except Exception as e:
        error_msg = f"ERRO ao executar treinamento: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return False


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Executa grid de experimentos de forma idempotente'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Diretório base dos dados comprimidos'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='out',
        help='Diretório de saída para modelos'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['MobileNetV2', 'VGG16', 'VGG19'],
        help='Modelos para treinar'
    )
    parser.add_argument(
        '--train_qualities',
        type=int,
        nargs='+',
        default=[1, 5, 10],
        help='Qualidades de treino'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[42, 123, 456, 789, 1000],
        help='Seeds para replicações'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Número de épocas'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Tamanho do batch'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Paciência para early stopping'
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Pula experimentos que já existem no CSV'
    )
    
    args = parser.parse_args()
    
    # Cria diretórios
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Caminho do CSV
    csv_path = output_dir.parent / 'tabela_resultados.csv'
    
    print("="*60)
    print("EXECUÇÃO DE GRID DE EXPERIMENTOS")
    print("="*60)
    print(f"Modelos: {args.models}")
    print(f"Train Qualities: {args.train_qualities}")
    print(f"Seeds: {args.seeds}")
    print(f"CSV: {csv_path}")
    print(f"Skip existing: {args.skip_existing}")
    print("="*60 + "\n")
    
    total_experiments = len(args.models) * len(args.train_qualities) * len(args.seeds)
    completed = 0
    skipped = 0
    failed = 0
    
    # Executa grid
    for model in args.models:
        for train_q in args.train_qualities:
            for seed in args.seeds:
                # Verifica se já existe
                if args.skip_existing and check_if_already_run(csv_path, model, train_q, seed):
                    print(f"Pulando {model} q{train_q} seed{seed} (já existe no CSV)")
                    skipped += 1
                    continue
                
                # Configura logging
                logger = setup_logging(log_dir, model, train_q, seed)
                
                # Executa treinamento
                success = run_training(
                    model=model,
                    train_quality=train_q,
                    seed=seed,
                    data_dir=args.data_dir,
                    output_dir=str(output_dir),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    patience=args.patience,
                    logger=logger
                )
                
                if success:
                    completed += 1
                else:
                    failed += 1
                
                print(f"\nProgresso: {completed + skipped + failed}/{total_experiments} "
                      f"(completos: {completed}, pulados: {skipped}, falhas: {failed})\n")
    
    # Resumo final
    print("="*60)
    print("RESUMO FINAL")
    print("="*60)
    print(f"Total de experimentos: {total_experiments}")
    print(f"Completos: {completed}")
    print(f"Pulados: {skipped}")
    print(f"Falhas: {failed}")
    print("="*60)


if __name__ == '__main__':
    main()
