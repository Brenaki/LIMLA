"""
Script para testar imagens individuais com modelo treinado.
"""

import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys

# Adiciona diretório raiz ao path para importar módulos
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infrastructure.model_builder import build_model
from src.infrastructure.checkpoint import load_checkpoint


def load_classes(classes_path: Path) -> dict:
    """
    Carrega mapeamento de classes do arquivo JSON.
    
    Args:
        classes_path: Caminho do arquivo classes.json
        
    Returns:
        Dicionário {índice: nome_classe}
    """
    if not classes_path.exists():
        raise FileNotFoundError(f"Arquivo de classes não encontrado: {classes_path}")
    
    with open(classes_path, 'r') as f:
        return json.load(f)


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Pré-processa imagem para inferência.
    
    Args:
        image_path: Caminho da imagem
        
    Returns:
        Tensor da imagem pré-processada
    """
    # Transformações (mesmas da validação)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Carrega e processa imagem
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Adiciona dimensão batch
    
    return image_tensor


def predict_image(
    model: nn.Module,
    image_path: str,
    device: torch.device,
    classes: dict
) -> tuple:
    """
    Faz predição em uma imagem.
    
    Args:
        model: Modelo treinado
        image_path: Caminho da imagem
        device: Dispositivo (CPU ou GPU)
        classes: Dicionário de classes {índice: nome}
        
    Returns:
        Tupla (classe_predita, probabilidades_dict)
    """
    # Pré-processa imagem
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Faz predição
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
    
    # Converte probabilidades para dict
    probs_dict = {
        classes.get(str(i), f"Classe_{i}"): prob.item()
        for i, prob in enumerate(probabilities[0])
    }
    
    predicted_class = classes.get(str(predicted_idx), f"Classe_{predicted_idx}")
    
    return predicted_class, probs_dict


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Testa uma imagem com modelo treinado',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python scripts/test_image.py --model_path out/MobileNetV2/best.pt --image_path test.jpg
  python scripts/test_image.py --model_path out/VGG16/best.pt --image_path test.jpg --classes_file out/VGG16/classes.json
        """
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Caminho para o modelo treinado (.pt)'
    )
    
    parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='Caminho para a imagem a testar'
    )
    
    parser.add_argument(
        '--classes_file',
        type=str,
        default=None,
        help='Caminho para arquivo classes.json (opcional, tenta detectar automaticamente)'
    )
    
    args = parser.parse_args()
    
    # Valida arquivos
    model_path = Path(args.model_path)
    if not model_path.exists():
        parser.error(f"Modelo não encontrado: {model_path}")
    
    image_path = Path(args.image_path)
    if not image_path.exists():
        parser.error(f"Imagem não encontrada: {image_path}")
    
    # Detecta arquivo de classes
    if args.classes_file:
        classes_path = Path(args.classes_file)
    else:
        # Tenta encontrar classes.json no mesmo diretório do modelo
        classes_path = model_path.parent / 'classes.json'
    
    if not classes_path.exists():
        parser.error(
            f"Arquivo de classes não encontrado: {classes_path}. "
            f"Use --classes_file para especificar o caminho."
        )
    
    # Carrega classes
    print(f"Carregando classes de: {classes_path}")
    classes = load_classes(classes_path)
    num_classes = len(classes)
    print(f"Classes carregadas ({num_classes}): {', '.join(classes.values())}\n")
    
    # Configura dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}\n")
    
    # Carrega modelo
    print(f"Carregando modelo de: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Detecta nome do modelo a partir do caminho ou tenta ambos
    model_dir_name = model_path.parent.name
    if 'mobilenet' in model_dir_name.lower() or 'mobilenet' in str(model_path).lower():
        model_name = 'MobileNetV2'
    elif 'vgg' in model_dir_name.lower() or 'vgg' in str(model_path).lower():
        model_name = 'VGG16'
    else:
        # Tenta carregar ambos (pode ser lento, mas funciona)
        print("Tentando detectar modelo automaticamente...")
        try:
            model = build_model('MobileNetV2', num_classes, pretrained=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model_name = 'MobileNetV2'
        except:
            model = build_model('VGG16', num_classes, pretrained=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model_name = 'VGG16'
    
    # Constrói modelo
    model = build_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Modelo {model_name} carregado com sucesso!\n")
    
    # Faz predição
    print(f"Processando imagem: {image_path}")
    predicted_class, probabilities = predict_image(model, str(image_path), device, classes)
    
    # Exibe resultados
    print("\n" + "="*60)
    print("RESULTADO DA PREDIÇÃO")
    print("="*60)
    print(f"\nClasse predita: {predicted_class}")
    print(f"\nProbabilidades:")
    for cls, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {prob*100:.2f}%")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
