# LIMLA - Lossy Impact on Machine Learning Algorithms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![wakatime](https://wakatime.com/badge/user/fdb75ce3-7263-4128-b59f-498eaf060cb7/project/ff82487a-5423-4677-8f63-fa51816fa944.svg)](https://wakatime.com/badge/user/fdb75ce3-7263-4128-b59f-498eaf060cb7/project/ff82487a-5423-4677-8f63-fa51816fa944)

## 📋 About the Project

**LIMLA** (Lossy Impact on Machine Learning Algorithms) is a research project that systematically investigates the impact of **lossy compression** on digital media on the performance of Machine Learning (ML) models, especially Deep Learning (DL).

The growing use of Machine Learning algorithms to process digital media (images, audio, and video) is fundamental in areas such as healthcare, security, and automotive. However, the large volume of this data often requires the use of lossy compression to optimize storage and transmission, especially in resource-constrained environments.

Lossy compression, while efficient in reducing file sizes, introduces artifacts and degradations that can severely compromise ML model performance. This project aims to:

- ✅ Evaluate the performance of Deep Learning models in classification tasks using media subjected to different levels of compression severity
- ✅ Compare the impact of compression degradations on different types of media
- ✅ Investigate the effect of lossy compression on deep neural networks during training and inference
- ✅ Analyze the interaction between lossy compression and adversarial attacks

## 🎯 Objectives

### General Objective
Systematically investigate the impact of lossy compression on digital media on ML model performance, considering compression severity, different types of media, neural network architectures, and their interaction in scenarios with adversarial attacks.

### Specific Objectives
1. Evaluate the performance of DL models in specific tasks (classification and detection) using media subjected to different levels of compression severity
2. Compare the impact of compression degradations on different types of media (images, audio, and video)
3. Investigate the effect of lossy compression on deep neural networks during training and inference phases
4. Analyze the interaction between lossy compression and adversarial attacks

## 🏗️ Project Architecture

The project consists of two main components:

### 1. Compression Pipeline (Rust)
A system developed in Rust that processes image datasets, applies JPEG compression with different quality levels, and organizes data for training.

**Features:**
- Parallel image processing using `rayon`
- Automatic dataset splitting into train/val/test
- JPEG compression with multiple quality levels (QF: 1-100)
- Hierarchical organization of compressed data
- Integration with CNN training pipeline

### 2. CNN Training System (Python)
A CNN model training system using PyTorch, following Clean Architecture.

**Features:**
- Support for multiple architectures (MobileNetV2, VGG16)
- Transfer learning with pre-trained models
- Configurable early stopping
- Automatic class detection
- Scripts for testing individual images

## 📁 Project Structure

```
LIMLA/
├── src/
│   └── main.rs              # Compression pipeline in Rust
├── cnn/                     # CNN training system
│   ├── src/
│   │   ├── domain/          # Domain layer (entities and interfaces)
│   │   ├── data/            # Data layer (datasets and dataloaders)
│   │   ├── use_cases/       # Use cases (training and early stopping)
│   │   ├── infrastructure/  # Infrastructure layer (models and checkpoints)
│   │   └── presentation/    # Presentation layer (CLI and progress)
│   ├── scripts/
│   │   └── test_image.py    # Individual image testing
│   ├── main.py              # Main entry point
│   └── README.md            # CNN module documentation
├── docs/
│   └── i18n/
│       └── pt/              # Portuguese (Brazilian) translations
├── Cargo.toml               # Rust dependencies
├── ROADMAP.md               # Project roadmap
├── subproject.md            # Research project documentation
└── README.md                # This file
```

## 🚀 Installation

### Prerequisites

- **Rust** 1.70 or higher
- **Python** 3.13 or higher
- **PyTorch** 2.0 or higher
- **CUDA** (optional, for GPU acceleration)

### Compression Pipeline Installation (Rust)

```bash
# Clone the repository
git clone https://github.com/Brenaki/LIMLA.git
cd LIMLA

# Build the project
cargo build --release
```

### CNN System Installation (Python)

```bash
# Enter the CNN directory
cd cnn

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## 💻 Usage

### Compression Pipeline

The pipeline processes an image dataset, applies compression, and organizes the data:

```bash
# Basic usage
cargo run --release -- \
    --path ./dataset \
    --quality "1,5,10" \
    --output ./compressed

# With custom split (train/val/test)
cargo run --release -- \
    --path ./dataset \
    --train 0.7 \
    --val 0.15 \
    --test 0.15 \
    --quality "1,5,10,20,50" \
    --output ./compressed

# With automatic training after compression
cargo run --release -- \
    --path ./dataset \
    --quality "1,5,10" \
    --output ./compressed \
    run --model MobileNetV2 \
        --epochs 50 \
        --batch_size 8 \
        --patience 5
```

**Parameters:**
- `--path`: Path to the dataset (structure: `dataset/class1/*.jpeg`, `dataset/class2/*.jpeg`)
- `--train`, `--val`, `--test`: Split percentages (default: 0.8, 0.1, 0.1)
- `--quality`: JPEG quality levels separated by comma (default: "1,5,10")
- `--output`: Output directory (default: "./compressed")

### CNN Model Training

```bash
# Train MobileNetV2 with quality 1
cd cnn
python main.py \
    --model MobileNetV2 \
    --data_dir ../compressed \
    --quality 1 \
    --epochs 50 \
    --batch_size 32 \
    --patience 5 \
    --output_dir ./out

# Train VGG16 with quality 5
python main.py \
    --model VGG16 \
    --data_dir ../compressed \
    --quality 5 \
    --epochs 100 \
    --batch_size 16 \
    --patience 10 \
    --learning_rate 0.0001
```

**Parameters:**
- `--model`: Model to train (MobileNetV2 or VGG16)
- `--data_dir`: Base directory of compressed data
- `--quality`: Image quality (1, 5, 10, etc.)
- `--epochs`: Maximum number of epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--patience`: Epochs without improvement for early stopping (default: 5)
- `--output_dir`: Output directory (default: "out/")

### Test Individual Image

```bash
cd cnn
python scripts/test_image.py \
    --model_path out/MobileNetV2/best.pt \
    --image_path my_image.jpg
```

## 📊 Data Structure

The pipeline generates the following directory structure:

```
compressed/
├── q1/                    # Quality 1
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
├── q5/                    # Quality 5
└── q10/                   # Quality 10
```

## 🔬 Results and Metrics

The system evaluates model performance using the following metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Loss** (Training and Validation)

Trained models are saved with:
- Best model (lowest val_loss)
- Last checkpoint
- Class mapping
- Best model information (JSON)

## 🛠️ Technologies Used

### Compression Pipeline
- **Rust** - Programming language
- **rayon** - Parallel processing
- **image** - Image processing
- **clap** - Command-line interface
- **indicatif** - Progress bars

### CNN System
- **Python 3.13+**
- **PyTorch** - Deep Learning (DL) framework
- **torchvision** - Pre-trained models and transformations
- **tqdm** - Progress bars
- **Pillow** - Image processing
- **NumPy** - Numerical computation

## 📈 Methodology

The project follows a systematic methodology:

1. **Database Selection**: Survey of publicly recognized databases as ML benchmarks
2. **Compression Application**: Use of lossy compression algorithms (JPEG, JPEG2000, H.264, HEVC, MP3, Opus) with multiple levels
3. **Model Training**: Selection and training of deep neural network architectures suitable for each media type and task
4. **Quantitative Evaluation**: Robustness analysis using appropriate metrics (accuracy, precision, recall, F1-score)

## 📝 References

This project is based on academic research on the impact of lossy compression on Machine Learning (ML) models. For more details on the theoretical foundation, see `subproject.md`.

## 🌐 Translations

Documentation is available in other languages:

| Language | Path |
|----------|------|
| Português Brasileiro | [docs/i18n/pt/](docs/i18n/pt/) |

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Victor Cerqueira**

- Email: victor.legat.cerqueira@gmail.com
- GitHub: [@Brenaki](https://github.com/Brenaki)

## 🙏 Acknowledgments

This project is part of academic research on the impact of lossy compression on Machine Learning (ML) algorithms. We thank the open source community for the tools and libraries that made this project possible.

---

**Note**: This is an actively developing project. Results and features may be updated as the research progresses.
