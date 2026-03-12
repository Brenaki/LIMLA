"""
Interface de linha de comando (CLI) usando argparse.
"""

import argparse
from pathlib import Path
from ..domain.models import SUPPORTED_MODELS
from ..data.quality_paths import resolve_quality_split_dir, validate_quality_value


def parse_args():
    """
    Parse argumentos da linha de comando.
    
    Returns:
        Namespace com argumentos parseados
    """
    parser = argparse.ArgumentParser(
        description='Treina modelos CNN (MobileNetV2, VGG16) usando PyTorch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python main.py --model MobileNetV2 --data_dir ./compressed --quality 1 --epochs 50
  python main.py --model VGG16 --data_dir ./compressed --quality 5 --epochs 100 --patience 10
        """
    )
    
    # Modelo
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=SUPPORTED_MODELS,
        help=f'Modelo a ser treinado: {", ".join(SUPPORTED_MODELS)}'
    )
    
    # Dados
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Diretório base dos dados comprimidos (ex: ./compressed)'
    )
    
    parser.add_argument(
        '--quality',
        type=str,
        default='1',
        help="Qualidade da imagem (1-100) ou 'original' - padrão: 1"
    )
    
    # Treinamento
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Número máximo de épocas (padrão: 50)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Tamanho do batch (padrão: 32)'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Taxa de aprendizado (padrão: 0.001)'
    )
    
    # Early stopping
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Número de épocas sem melhoria para early stopping (padrão: 5)'
    )
    
    parser.add_argument(
        '--min_delta',
        type=float,
        default=0.001,
        help='Melhoria mínima considerada como progresso (padrão: 0.001)'
    )
    
    # Saída
    parser.add_argument(
        '--output_dir',
        type=str,
        default='out',
        help='Diretório de saída para modelos treinados (padrão: out/)'
    )
    
    # Opcional
    parser.add_argument(
        '--num_classes',
        type=int,
        default=None,
        help='Número de classes (opcional, detecta automaticamente se não fornecido)'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Número de workers para carregamento de dados (padrão: 4)'
    )
    
    # Seeds e reprodutibilidade
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Seed para reprodutibilidade (opcional, padrão: None)'
    )
    
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Habilita modo determinístico (mais lento, mas totalmente reprodutível)'
    )
    
    # Resume/Checkpoint
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Caminho do checkpoint para continuar treinamento (ex: out/MobileNetV2_q10_seed42/last.pt). Se não especificado, tenta detectar automaticamente.'
    )
    
    parser.add_argument(
        '--checkpoint_interval',
        type=int,
        default=5,
        help='Intervalo em épocas para salvar checkpoints intermediários (padrão: 5)'
    )
    
    args = parser.parse_args()

    # Validações
    try:
        args.quality = validate_quality_value(args.quality)
    except ValueError as exc:
        parser.error(str(exc))
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        parser.error(f"Diretório de dados não encontrado: {args.data_dir}")
    
    # Valida estrutura de pastas
    try:
        train_dir = resolve_quality_split_dir(data_dir, args.quality, 'train')
        val_dir = resolve_quality_split_dir(data_dir, args.quality, 'val')
    except ValueError as exc:
        parser.error(str(exc))
    
    return args
