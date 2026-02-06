# LIMLA - Impacto da Compressão com Perdas em Algoritmos de Machine Learning

[![Licença: MIT](https://img.shields.io/badge/Licen%C3%A7a-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![wakatime](https://wakatime.com/badge/user/fdb75ce3-7263-4128-b59f-498eaf060cb7/project/ff82487a-5423-4677-8f63-fa51816fa944.svg)](https://wakatime.com/badge/user/fdb75ce3-7263-4128-b59f-498eaf060cb7/project/ff82487a-5423-4677-8f63-fa51816fa944)

## Sobre o Projeto

**LIMLA** (Lossy Impact on Machine Learning Algorithms) é um projeto de pesquisa que investiga sistematicamente o impacto da **compressão com perdas** em mídias digitais no desempenho de modelos de Machine Learning (ML), especialmente Deep Learning (DL).

O uso crescente de algoritmos de Machine Learning para processar mídias digitais (imagens, áudio e vídeo) é fundamental em áreas como saúde, segurança e automotiva. No entanto, o grande volume desses dados frequentemente exige o uso de compressão com perdas para otimizar armazenamento e transmissão, especialmente em ambientes com recursos limitados.

A compressão com perdas, embora eficiente em reduzir o tamanho dos arquivos, introduz artefatos e degradações que podem comprometer severamente o desempenho dos modelos de ML. Este projeto visa:

- Avaliar o desempenho de modelos de Deep Learning em tarefas de classificação usando mídias submetidas a diferentes níveis de severidade de compressão
- Comparar o impacto das degradações da compressão em diferentes tipos de mídia
- Investigar o efeito da compressão com perdas em redes neurais profundas durante treinamento e inferência
- Analisar a interação entre compressão com perdas e ataques adversariais

## Objetivos

### Objetivo Geral
Investigar sistematicamente o impacto da compressão com perdas em mídias digitais no desempenho de modelos de ML, considerando a severidade da compressão, diferentes tipos de mídia, arquiteturas de redes neurais e sua interação em cenários com ataques adversariais.

### Objetivos Específicos
1. Avaliar o desempenho de modelos de DL em tarefas específicas (classificação e detecção) usando mídias submetidas a diferentes níveis de severidade de compressão
2. Comparar o impacto das degradações da compressão em diferentes tipos de mídia (imagens, áudio e vídeo)
3. Investigar o efeito da compressão com perdas em redes neurais profundas durante as fases de treinamento e inferência
4. Analisar a interação entre compressão com perdas e ataques adversariais

## Arquitetura do Projeto

O projeto consiste em dois componentes principais:

### 1. Pipeline de Compressão (Rust)
Sistema desenvolvido em Rust que processa datasets de imagens, aplica compressão JPEG com diferentes níveis de qualidade e organiza os dados para treinamento.

**Recursos:**
- Processamento paralelo de imagens usando `rayon`
- Divisão automática do dataset em train/val/test
- Compressão JPEG com múltiplos níveis de qualidade (QF: 1-100)
- Organização hierárquica dos dados comprimidos
- Integração com o pipeline de treinamento CNN

### 2. Sistema de Treinamento CNN (Python)
Sistema de treinamento de modelos CNN usando PyTorch, seguindo Clean Architecture.

**Recursos:**
- Suporte a múltiplas arquiteturas (MobileNetV2, VGG16)
- Transfer learning com modelos pré-treinados
- Early stopping configurável
- Detecção automática de classes
- Scripts para teste de imagens individuais

## Estrutura do Projeto

```
LIMLA/
├── src/
│   └── main.rs              # Pipeline de compressão em Rust
├── cnn/                     # Sistema de treinamento CNN
│   ├── src/
│   │   ├── domain/          # Camada de domínio (entidades e interfaces)
│   │   ├── data/            # Camada de dados (datasets e dataloaders)
│   │   ├── use_cases/       # Casos de uso (treinamento e early stopping)
│   │   ├── infrastructure/  # Camada de infraestrutura (modelos e checkpoints)
│   │   └── presentation/    # Camada de apresentação (CLI e progresso)
│   ├── scripts/
│   │   └── test_image.py    # Teste de imagem individual
│   ├── main.py              # Ponto de entrada principal
│   └── README.md            # Documentação do módulo CNN
├── Cargo.toml               # Dependências Rust
├── subproject.md            # Documentação do projeto de pesquisa
└── README.md                # Este arquivo
```

## Instalação

### Pré-requisitos

- **Rust** 1.70 ou superior
- **Python** 3.13 ou superior
- **PyTorch** 2.0 ou superior
- **CUDA** (opcional, para aceleração em GPU)

### Instalação do Pipeline de Compressão (Rust)

```bash
# Clone o repositório
git clone https://github.com/Brenaki/LIMLA.git
cd LIMLA

# Compile o projeto
cargo build --release
```

### Instalação do Sistema CNN (Python)

```bash
# Entre no diretório CNN
cd cnn

# Instale as dependências usando uv (recomendado)
uv sync

# Ou usando pip
pip install -e .
```

## Uso

### Pipeline de Compressão

O pipeline processa um dataset de imagens, aplica compressão e organiza os dados:

```bash
# Uso básico
cargo run --release -- \
    --path ./dataset \
    --quality "1,5,10" \
    --output ./compressed

# Com divisão customizada (train/val/test)
cargo run --release -- \
    --path ./dataset \
    --train 0.7 \
    --val 0.15 \
    --test 0.15 \
    --quality "1,5,10,20,50" \
    --output ./compressed

# Com treinamento automático após compressão
cargo run --release -- \
    --path ./dataset \
    --quality "1,5,10" \
    --output ./compressed \
    run --model MobileNetV2 \
        --epochs 50 \
        --batch_size 8 \
        --patience 5
```

**Parâmetros:**
- `--path`: Caminho para o dataset (estrutura: `dataset/classe1/*.jpeg`, `dataset/classe2/*.jpeg`)
- `--train`, `--val`, `--test`: Percentuais de divisão (padrão: 0.8, 0.1, 0.1)
- `--quality`: Níveis de qualidade JPEG separados por vírgula (padrão: "1,5,10")
- `--output`: Diretório de saída (padrão: "./compressed")

### Treinamento do Modelo CNN

```bash
# Treinar MobileNetV2 com qualidade 1
cd cnn
python main.py \
    --model MobileNetV2 \
    --data_dir ../compressed \
    --quality 1 \
    --epochs 50 \
    --batch_size 32 \
    --patience 5 \
    --output_dir ./out

# Treinar VGG16 com qualidade 5
python main.py \
    --model VGG16 \
    --data_dir ../compressed \
    --quality 5 \
    --epochs 100 \
    --batch_size 16 \
    --patience 10 \
    --learning_rate 0.0001
```

**Parâmetros:**
- `--model`: Modelo a treinar (MobileNetV2 ou VGG16)
- `--data_dir`: Diretório base dos dados comprimidos
- `--quality`: Qualidade da imagem (1, 5, 10, etc.)
- `--epochs`: Número máximo de épocas (padrão: 50)
- `--batch_size`: Tamanho do batch (padrão: 32)
- `--learning_rate`: Taxa de aprendizado (padrão: 0.001)
- `--patience`: Épocas sem melhoria para early stopping (padrão: 5)
- `--output_dir`: Diretório de saída (padrão: "out/")

### Testar Imagem Individual

```bash
cd cnn
python scripts/test_image.py \
    --model_path out/MobileNetV2/best.pt \
    --image_path minha_imagem.jpg
```

## Estrutura de Dados

O pipeline gera a seguinte estrutura de diretórios:

```
compressed/
├── q1/                    # Qualidade 1
│   ├── train/
│   │   ├── class1/
│   │   │   ├── img1.jpg
│   │   │   └── img2.jpg
│   │   └── class2/
│   │       └── ...
│   ├── val/
│   │   ├── class1/
│   │   └── class2/
│   └── test/
│       ├── class1/
│       └── class2/
├── q5/                    # Qualidade 5
└── q10/                   # Qualidade 10
```

## Resultados e Métricas

O sistema avalia o desempenho do modelo usando as seguintes métricas:

- **Acurácia**
- **Precisão**
- **Recall**
- **F1-Score**
- **Loss** (Treinamento e Validação)

Os modelos treinados são salvos com:
- Melhor modelo (menor val_loss)
- Último checkpoint
- Mapeamento de classes
- Informações do melhor modelo (JSON)

## Tecnologias Utilizadas

### Pipeline de Compressão
- **Rust** - Linguagem de programação
- **rayon** - Processamento paralelo
- **image** - Processamento de imagens
- **clap** - Interface de linha de comando
- **indicatif** - Barras de progresso

### Sistema CNN
- **Python 3.13+**
- **PyTorch** - Framework de Deep Learning (DL)
- **torchvision** - Modelos pré-treinados e transformações
- **tqdm** - Barras de progresso
- **Pillow** - Processamento de imagens
- **NumPy** - Computação numérica

## Metodologia

O projeto segue uma metodologia sistemática:

1. **Seleção de Bases de Dados**: Levantamento de bases públicas reconhecidas como benchmarks em ML
2. **Aplicação de Compressão**: Uso de algoritmos de compressão com perdas (JPEG, JPEG2000, H.264, HEVC, MP3, Opus) com múltiplos níveis
3. **Treinamento de Modelos**: Seleção e treinamento de arquiteturas de redes neurais profundas adequadas a cada tipo de mídia e tarefa
4. **Avaliação Quantitativa**: Análise de robustez usando métricas apropriadas (acurácia, precisão, recall, F1-score)

## Referências

Este projeto é baseado em pesquisa acadêmica sobre o impacto da compressão com perdas em modelos de Machine Learning (ML). Para mais detalhes sobre a fundamentação teórica, consulte `subproject.md`.

## Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:

1. Fazer fork do projeto
2. Criar uma branch de feature (`git checkout -b feature/MinhaFeature`)
3. Fazer commit das alterações (`git commit -m 'Adiciona MinhaFeature'`)
4. Enviar para a branch (`git push origin feature/MinhaFeature`)
5. Abrir um Pull Request

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](../../../LICENSE) para detalhes.

## Autor

**Victor Cerqueira**

- Email: victor.legat.cerqueira@gmail.com
- GitHub: [@Brenaki](https://github.com/Brenaki)

## Agradecimentos

Este projeto faz parte de uma pesquisa acadêmica sobre o impacto da compressão com perdas em algoritmos de Machine Learning (ML). Agradecemos à comunidade open source pelas ferramentas e bibliotecas que tornaram este projeto possível.

---

**Nota**: Este é um projeto em desenvolvimento ativo. Resultados e funcionalidades podem ser atualizados conforme a pesquisa avança.
