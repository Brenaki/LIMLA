"""
Visualizações dos resultados de treinamento.
Gera gráficos principais focados em robustez no teste.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Optional


def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple:
    """
    Calcula intervalo de confiança usando distribuição t.
    
    Args:
        data: Array com dados
        confidence: Nível de confiança (padrão: 0.95)
        
    Returns:
        Tupla (média, erro padrão, limite_inferior, limite_superior)
    """
    if len(data) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Desvio padrão amostral
    n = len(data)
    
    # Erro padrão
    se = std / np.sqrt(n) if n > 0 else 0
    
    # Valor crítico da distribuição t
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1) if n > 1 else 1.96
    
    # Intervalo de confiança
    ci_lower = mean - t_critical * se
    ci_upper = mean + t_critical * se
    
    return (mean, se, ci_lower, ci_upper)


def fix_test_quality_order(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fixa ordem dos níveis de test_quality.
    Ordem: original → 50 → 20 → 10 → 5 → 1
    
    Args:
        data: DataFrame
        
    Returns:
        DataFrame com ordem fixa
    """
    if 'test_quality' not in data.columns:
        return data
    
    # Define ordem
    quality_order = ['original', '50', '20', '10', '5', '1']
    
    # Converte para categoria ordenada
    data['test_quality'] = pd.Categorical(
        data['test_quality'].astype(str),
        categories=quality_order,
        ordered=True
    )
    
    return data


def plot_test_accuracy_vs_quality(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int = 300
) -> None:
    """
    Gráfico principal: test accuracy vs test_quality (linha com IC 95%).
    
    Args:
        df: DataFrame com dados filtrados
        output_path: Caminho para salvar figura
        dpi: Resolução da figura
    """
    # Filtra dados
    filtered = df[
        (df['split'] == 'test') &
        (df['epoch_type'] == 'best') &
        (df['metric'] == 'accuracy')
    ].copy()
    
    filtered['value'] = pd.to_numeric(filtered['value'], errors='coerce')
    filtered = filtered.dropna(subset=['value'])
    filtered = fix_test_quality_order(filtered)
    
    # Prepara dados para plot
    models = filtered['model'].unique()
    train_qualities = sorted(filtered['train_quality'].unique())
    
    # Cria figura com subplots
    n_facets = len(train_qualities)
    fig, axes = plt.subplots(1, n_facets, figsize=(6*n_facets, 5))
    if n_facets == 1:
        axes = [axes]
    
    # Cores para modelos
    colors = sns.color_palette("husl", len(models))
    model_colors = {model: colors[i] for i, model in enumerate(models)}
    
    for idx, train_q in enumerate(train_qualities):
        ax = axes[idx]
        
        for model in models:
            subset = filtered[
                (filtered['model'] == model) &
                (filtered['train_quality'] == train_q)
            ]
            
            if len(subset) == 0:
                continue
            
            # Agrupa por test_quality e calcula IC
            test_qualities = sorted(subset['test_quality'].cat.categories)
            means = []
            ci_lowers = []
            ci_uppers = []
            valid_qualities = []
            
            for tq in test_qualities:
                if tq not in subset['test_quality'].values:
                    continue
                
                values = subset[subset['test_quality'] == tq]['value'].values
                if len(values) > 0:
                    mean, se, ci_lower, ci_upper = calculate_confidence_interval(values)
                    means.append(mean)
                    ci_lowers.append(ci_lower)
                    ci_uppers.append(ci_upper)
                    valid_qualities.append(tq)
            
            if len(means) > 0:
                # Plota linha com IC
                ax.plot(valid_qualities, means, marker='o', label=model,
                       color=model_colors[model], linewidth=2, markersize=8)
                ax.fill_between(valid_qualities, ci_lowers, ci_uppers,
                               alpha=0.2, color=model_colors[model])
        
        ax.set_xlabel('Test Quality', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title(f'Train Quality = {train_q}', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Gráfico salvo em: {output_path}")
    plt.close()


def plot_test_accuracy_distribution(
    df: pd.DataFrame,
    output_path: Path,
    dpi: int = 300
) -> None:
    """
    Boxplot/Violin: distribuição de accuracy por test_quality.
    
    Args:
        df: DataFrame com dados filtrados
        output_path: Caminho para salvar figura
        dpi: Resolução da figura
    """
    # Filtra dados
    filtered = df[
        (df['split'] == 'test') &
        (df['epoch_type'] == 'best') &
        (df['metric'] == 'accuracy')
    ].copy()
    
    filtered['value'] = pd.to_numeric(filtered['value'], errors='coerce')
    filtered = filtered.dropna(subset=['value'])
    filtered = fix_test_quality_order(filtered)
    
    # Prepara dados
    train_qualities = sorted(filtered['train_quality'].unique())
    n_facets = len(train_qualities)
    
    fig, axes = plt.subplots(1, n_facets, figsize=(6*n_facets, 5))
    if n_facets == 1:
        axes = [axes]
    
    for idx, train_q in enumerate(train_qualities):
        ax = axes[idx]
        
        subset = filtered[filtered['train_quality'] == train_q]
        
        # Boxplot
        sns.boxplot(
            data=subset,
            x='test_quality',
            y='value',
            hue='model',
            ax=ax
        )
        
        ax.set_xlabel('Test Quality', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title(f'Train Quality = {train_q}', fontsize=14, fontweight='bold')
        ax.legend(title='Model', loc='best')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Gráfico salvo em: {output_path}")
    plt.close()


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Gera visualizações dos resultados'
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        default='../tabela_resultados.csv',
        help='Caminho para o CSV de resultados'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../results',
        help='Diretório de saída para figuras'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Resolução das figuras (padrão: 300)'
    )
    
    args = parser.parse_args()
    
    # Cria diretórios de saída
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)
    
    # Carrega dados
    print(f"Carregando dados de: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"Total de linhas: {len(df)}")
    
    # Configura estilo
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = args.dpi
    plt.rcParams['savefig.dpi'] = args.dpi
    
    # Gera gráficos
    print("\nGerando gráficos...")
    
    # Gráfico 1: Linha com IC
    print("1. Gráfico: Test Accuracy vs Test Quality (linha com IC 95%)")
    plot_test_accuracy_vs_quality(
        df,
        output_dir / 'figures' / 'test_accuracy_vs_quality.png',
        dpi=args.dpi
    )
    
    # Gráfico 2: Boxplot
    print("2. Gráfico: Distribuição de Test Accuracy (boxplot)")
    plot_test_accuracy_distribution(
        df,
        output_dir / 'figures' / 'test_accuracy_distribution.png',
        dpi=args.dpi
    )
    
    print("\nVisualizações concluídas!")


if __name__ == '__main__':
    main()
