# Sistema de Treinamento CNN com PyTorch

Sistema de treinamento de modelos CNN (MobileNetV2 e VGG16) usando PyTorch, seguindo Clean Architecture.

## Estrutura do Projeto

```
cnn/
├── src/
│   ├── domain/           # Camada de domínio (entidades e interfaces)
│   ├── data/             # Camada de dados (datasets e dataloaders)
│   ├── use_cases/        # Casos de uso (treinamento e early stopping)
│   ├── infrastructure/   # Camada de infraestrutura (modelos e checkpoints)
│   └── presentation/     # Camada de apresentação (CLI e progresso)
├── scripts/
│   └── test_image.py     # Script para testar imagens individuais
├── main.py               # Ponto de entrada principal
└── pyproject.toml        # Dependências
```

## Instalação

```bash
# Instalar dependências usando uv (recomendado)
uv sync

# Ou usando pip
pip install -e .
```

## Estrutura de Dados

O sistema espera dados organizados pela estrutura gerada pelo script Rust:

```
compressed/
├── q1/                    # Qualidade 1
│   ├── train/
│   │   ├── classe1/
│   │   │   ├── img1.jpg
│   │   │   └── img2.jpg
│   │   └── classe2/
│   │       └── ...
│   ├── val/
│   │   ├── classe1/
│   │   └── classe2/
│   └── test/ (opcional)
├── q5/                    # Qualidade 5
└── q10/                   # Qualidade 10
```

## Uso

### Treinamento

```bash
# Treinar MobileNetV2 com qualidade 1
python main.py --model MobileNetV2 --data_dir ./compressed --quality 1 --epochs 50

# Treinar VGG16 com qualidade 5 e early stopping personalizado
python main.py --model VGG16 --data_dir ./compressed --quality 5 --epochs 100 --patience 10 --min_delta 0.001

# Com parâmetros personalizados
python main.py \
    --model MobileNetV2 \
    --data_dir ./compressed \
    --quality 1 \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --patience 5 \
    --min_delta 0.001 \
    --output_dir out
```

### Testar Imagem Individual

```bash
# Testar uma imagem com modelo treinado
python scripts/test_image.py \
    --model_path out/MobileNetV2/best.pt \
    --image_path minha_imagem.jpg

# Especificar arquivo de classes manualmente
python scripts/test_image.py \
    --model_path out/VGG16/best.pt \
    --image_path minha_imagem.jpg \
    --classes_file out/VGG16/classes.json
```

### Gerar Relatório de Erros (HTML)

```bash
# Gerar relatório apenas com erros da validação
python scripts/generate_error_report.py \
    --model_path out/MobileNetV2/best.pt \
    --data_dir ./compressed \
    --quality 1

# Incluir também erros do treinamento
python scripts/generate_error_report.py \
    --model_path out/VGG16/best.pt \
    --data_dir ./compressed \
    --quality 5 \
    --include_train \
    --output relatorio_erros.html
```

## Parâmetros de Treinamento

- `--model`: Modelo a ser treinado (MobileNetV2 ou VGG16)
- `--data_dir`: Diretório base dos dados comprimidos
- `--quality`: Qualidade da imagem (1, 5, ou 10)
- `--epochs`: Número máximo de épocas (padrão: 50)
- `--batch_size`: Tamanho do batch (padrão: 32)
- `--learning_rate`: Taxa de aprendizado (padrão: 0.001)
- `--patience`: Épocas sem melhoria para early stopping (padrão: 5)
- `--min_delta`: Melhoria mínima considerada (padrão: 0.001)
- `--output_dir`: Diretório de saída (padrão: out/)
- `--num_workers`: Workers para carregamento de dados (padrão: 4)

## Saída

Os modelos treinados são salvos em:

```
out/
└── {model_name}/
    ├── best.pt          # Melhor modelo (menor val_loss)
    ├── last.pt          # Último checkpoint
    ├── classes.json     # Mapeamento de classes
    └── info.json        # Informações do melhor modelo
```

## Características

- ✅ Clean Architecture com separação clara de responsabilidades
- ✅ Suporte a MobileNetV2 e VGG16
- ✅ Early stopping configurável
- ✅ Progresso visual durante treinamento (tqdm)
- ✅ Detecção automática de classes
- ✅ Transfer learning com modelos pré-treinados
- ✅ Script para testar imagens individuais
- ✅ Script para gerar relatório HTML com imagens classificadas incorretamente
- ✅ Código didático e bem comentado em português
