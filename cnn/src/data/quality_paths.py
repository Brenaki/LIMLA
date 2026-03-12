"""
Resolução de caminhos de datasets por qualidade.
"""

from pathlib import Path
from typing import Optional


QualityValue = int | str


def normalize_quality_value(quality: QualityValue) -> QualityValue:
    """
    Normaliza o valor de qualidade recebido da CLI ou de scripts.

    Args:
        quality: Valor bruto da qualidade

    Returns:
        int para qualidades numéricas, ou 'original'
    """
    if isinstance(quality, str):
        value = quality.strip().lower()
        if value == 'original':
            return 'original'
        if value.startswith('q') and value[1:].isdigit():
            return int(value[1:])
        if value.isdigit():
            return int(value)
        return value
    return quality


def validate_quality_value(quality: QualityValue) -> QualityValue:
    """
    Valida qualidade numérica ou 'original'.
    """
    normalized_quality = normalize_quality_value(quality)

    if normalized_quality == 'original':
        return normalized_quality

    if not isinstance(normalized_quality, int):
        raise ValueError(
            f"Qualidade inválida: {quality}. Use um inteiro entre 1 e 100 ou 'original'."
        )

    if normalized_quality < 1 or normalized_quality > 100:
        raise ValueError(
            f"Qualidade deve estar entre 1 e 100, recebido: {normalized_quality}"
        )

    return normalized_quality


def quality_tag(quality: QualityValue) -> str:
    """
    Retorna uma tag estável para nomes de diretórios e runs.
    """
    normalized_quality = validate_quality_value(quality)
    if normalized_quality == 'original':
        return 'original'
    return f"q{normalized_quality}"


def _find_original_dir(data_dir: Path, split: Optional[str] = None) -> Optional[Path]:
    """
    Procura um diretório com imagens originais dentro ou ao lado de data_dir.

    Args:
        data_dir: Diretório base dos dados
        split: Split obrigatório dentro do diretório (opcional)

    Returns:
        Caminho do diretório original, se encontrado
    """
    candidates = []

    preferred_local = data_dir / 'original'
    preferred_sibling = data_dir.parent / 'original'

    if preferred_local.exists():
        candidates.append(preferred_local)
    if preferred_sibling.exists() and preferred_sibling != preferred_local:
        candidates.append(preferred_sibling)

    if data_dir.exists():
        dynamic_candidates = sorted(
            [
                item for item in data_dir.iterdir()
                if item.is_dir() and not item.name.startswith('q') and item.name != 'original'
            ],
            key=lambda item: item.name
        )
        candidates.extend(dynamic_candidates)

    for candidate in candidates:
        if split is None or (candidate / split).exists():
            return candidate

    return None


def resolve_quality_split_dir(
    data_dir: str | Path,
    quality: QualityValue,
    split: str
) -> Path:
    """
    Resolve o diretório de um split a partir da qualidade.

    Args:
        data_dir: Diretório base dos dados
        quality: Qualidade numérica ou 'original'
        split: Nome do split

    Returns:
        Caminho absoluto do split

    Raises:
        ValueError: Se o diretório correspondente não existir
    """
    data_dir_path = Path(data_dir)
    normalized_quality = validate_quality_value(quality)

    if normalized_quality == 'original':
        original_dir = _find_original_dir(data_dir_path, split=split)
        if original_dir:
            split_dir = original_dir / split
            if split_dir.exists():
                return split_dir

        raise ValueError(
            f"Diretório original não encontrado para split '{split}'. "
            f"Procurado em: {data_dir_path / 'original'} e {data_dir_path.parent / 'original'}."
        )

    split_dir = data_dir_path / quality_tag(normalized_quality) / split
    if split_dir.exists():
        return split_dir

    raise ValueError(
        f"Diretório não encontrado: {split_dir}. "
        f"Verifique se --data_dir e --quality estão corretos."
    )


def find_original_dir(data_dir: str | Path, split: Optional[str] = None) -> Optional[Path]:
    """
    Wrapper público para descoberta de diretório original.
    """
    return _find_original_dir(Path(data_dir), split=split)


def train_quality_sort_key(quality: QualityValue) -> tuple[int, int | str]:
    """
    Chave de ordenação para qualidades de treino.

    Ordem:
    - 'original' primeiro
    - qualidades numéricas em ordem crescente
    - outros valores string por último
    """
    normalized_quality = normalize_quality_value(quality)

    if normalized_quality == 'original':
        return (0, -1)

    if isinstance(normalized_quality, int):
        return (1, normalized_quality)

    return (2, str(normalized_quality))
