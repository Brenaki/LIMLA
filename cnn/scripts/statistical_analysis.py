"""
Análise estatística: ANOVA + Tukey HSD para resultados de treinamento.
Focada em test split e best checkpoint.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import f_oneway, levene
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.lib import qsturng
import json
from typing import Dict, List, Optional


def load_results_csv(csv_path: str) -> pd.DataFrame:
    """
    Carrega CSV de resultados.
    
    Args:
        csv_path: Caminho para o CSV
        
    Returns:
        DataFrame com resultados
    """
    df = pd.read_csv(csv_path)
    return df


def filter_data_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra dados para análise estatística.
    
    Filtros:
    - split == 'test'
    - epoch_type == 'best'
    - metric == 'accuracy'
    
    Args:
        df: DataFrame completo
        
    Returns:
        DataFrame filtrado
    """
    filtered = df[
        (df['split'] == 'test') &
        (df['epoch_type'] == 'best') &
        (df['metric'] == 'accuracy')
    ].copy()
    
    # Converte value para float
    filtered['value'] = pd.to_numeric(filtered['value'], errors='coerce')
    
    # Remove valores NaN
    filtered = filtered.dropna(subset=['value'])
    
    return filtered


def check_assumptions(data: pd.DataFrame, group_col: str, value_col: str = 'value') -> Dict:
    """
    Verifica suposições para ANOVA.
    
    Args:
        data: DataFrame com dados
        group_col: Nome da coluna com grupos
        value_col: Nome da coluna com valores
        
    Returns:
        Dicionário com resultados dos testes
    """
    groups = data[group_col].unique()
    group_data = [data[data[group_col] == g][value_col].values for g in groups]
    
    results = {}
    
    # Teste de normalidade (Shapiro-Wilk) - com cuidado: n pequeno é fraco
    normality_results = {}
    for group in groups:
        group_values = data[data[group_col] == group][value_col].values
        if len(group_values) >= 3:  # Mínimo para Shapiro-Wilk
            stat, p_value = stats.shapiro(group_values)
            normality_results[group] = {
                'statistic': stat,
                'p_value': p_value,
                'normal': p_value > 0.05
            }
        else:
            normality_results[group] = {
                'statistic': np.nan,
                'p_value': np.nan,
                'normal': None,
                'note': 'n < 3, teste não aplicável'
            }
    
    results['normality'] = normality_results
    
    # Teste de homocedasticidade (Levene)
    if len(group_data) >= 2 and all(len(g) >= 2 for g in group_data):
        levene_stat, levene_p = levene(*group_data)
        results['homogeneity'] = {
            'statistic': levene_stat,
            'p_value': levene_p,
            'homogeneous': levene_p > 0.05,
            'test': 'Levene'
        }
    else:
        results['homogeneity'] = {
            'statistic': np.nan,
            'p_value': np.nan,
            'homogeneous': None,
            'note': 'Dados insuficientes para teste'
        }
    
    return results


def one_way_anova(data: pd.DataFrame, factor: str, value_col: str = 'value') -> Dict:
    """
    Executa ANOVA one-way.
    
    Args:
        data: DataFrame com dados
        factor: Nome da coluna com fator (grupos)
        value_col: Nome da coluna com valores
        
    Returns:
        Dicionário com resultados da ANOVA
    """
    groups = data[factor].unique()
    group_data = [data[data[factor] == g][value_col].values for g in groups]
    
    # Remove grupos com menos de 2 observações
    valid_groups = []
    valid_data = []
    for g, d in zip(groups, group_data):
        if len(d) >= 2:
            valid_groups.append(g)
            valid_data.append(d)
    
    if len(valid_data) < 2:
        return {
            'error': 'Dados insuficientes para ANOVA (menos de 2 grupos válidos)'
        }
    
    # Executa ANOVA
    f_stat, p_value = f_oneway(*valid_data)
    
    # Calcula graus de liberdade
    n_total = sum(len(d) for d in valid_data)
    n_groups = len(valid_data)
    df_between = n_groups - 1
    df_within = n_total - n_groups
    
    # Calcula médias e variâncias
    means = {g: np.mean(d) for g, d in zip(valid_groups, valid_data)}
    stds = {g: np.std(d, ddof=1) for g, d in zip(valid_groups, valid_data)}
    
    return {
        'factor': factor,
        'groups': valid_groups,
        'f_statistic': float(f_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'df_between': int(df_between),
        'df_within': int(df_within),
        'n_total': int(n_total),
        'means': {str(k): float(v) for k, v in means.items()},
        'stds': {str(k): float(v) for k, v in stds.items()}
    }


def tukey_hsd(data: pd.DataFrame, factor: str, value_col: str = 'value', alpha: float = 0.05) -> pd.DataFrame:
    """
    Executa teste de Tukey HSD para comparações múltiplas.
    
    Args:
        data: DataFrame com dados
        factor: Nome da coluna com fator (grupos)
        value_col: Nome da coluna com valores
        alpha: Nível de significância
        
    Returns:
        DataFrame com resultados do Tukey
    """
    # Prepara dados para Tukey
    groups = data[factor].unique()
    
    # Remove grupos com menos de 2 observações
    valid_groups = []
    valid_data = []
    for g in groups:
        group_values = data[data[factor] == g][value_col].values
        if len(group_values) >= 2:
            valid_groups.append(g)
            valid_data.extend([(g, v) for v in group_values])
    
    if len(valid_groups) < 2:
        return pd.DataFrame()
    
    # Cria DataFrame para Tukey
    tukey_df = pd.DataFrame(valid_data, columns=[factor, value_col])
    
    # Executa Tukey HSD
    tukey_result = pairwise_tukeyhsd(
        endog=tukey_df[value_col],
        groups=tukey_df[factor],
        alpha=alpha
    )
    
    # Converte para DataFrame
    tukey_df_result = pd.DataFrame(data=tukey_result._results_table.data[1:],
                                   columns=tukey_result._results_table.data[0])
    
    return tukey_df_result


def games_howell(data: pd.DataFrame, factor: str, value_col: str = 'value', alpha: float = 0.05) -> pd.DataFrame:
    """
    Executa teste de Games-Howell (para variâncias heterogêneas).
    
    Args:
        data: DataFrame com dados
        factor: Nome da coluna com fator (grupos)
        value_col: Nome da coluna com valores
        alpha: Nível de significância
        
    Returns:
        DataFrame com resultados do Games-Howell
    """
    groups = data[factor].unique()
    
    # Calcula estatísticas por grupo
    group_stats = {}
    for g in groups:
        group_values = data[data[factor] == g][value_col].values
        if len(group_values) >= 2:
            group_stats[g] = {
                'mean': np.mean(group_values),
                'std': np.std(group_values, ddof=1),
                'n': len(group_values)
            }
    
    if len(group_stats) < 2:
        return pd.DataFrame()
    
    # Comparações pareadas
    comparisons = []
    group_names = list(group_stats.keys())
    
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            g1, g2 = group_names[i], group_names[j]
            stats1 = group_stats[g1]
            stats2 = group_stats[g2]
            
            # Diferença de médias
            mean_diff = stats1['mean'] - stats2['mean']
            
            # Erro padrão
            se = np.sqrt((stats1['std']**2 / stats1['n']) + (stats2['std']**2 / stats2['n']))
            
            # Graus de liberdade (Welch-Satterthwaite)
            df = (se**4) / (
                (stats1['std']**2 / stats1['n'])**2 / (stats1['n'] - 1) +
                (stats2['std']**2 / stats2['n'])**2 / (stats2['n'] - 1)
            )
            
            # Estatística t
            t_stat = mean_diff / se if se > 0 else 0
            
            # P-valor (distribuição t)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            # Correção de Bonferroni para múltiplas comparações
            n_comparisons = len(group_names) * (len(group_names) - 1) / 2
            p_adj = min(p_value * n_comparisons, 1.0)
            
            comparisons.append({
                'group1': str(g1),
                'group2': str(g2),
                'mean_diff': float(mean_diff),
                'se': float(se),
                't_stat': float(t_stat),
                'df': float(df),
                'p_value': float(p_value),
                'p_adj': float(p_adj),
                'reject': p_adj < alpha
            })
    
    return pd.DataFrame(comparisons)


def two_way_anova(data: pd.DataFrame, factor1: str, factor2: str, value_col: str = 'value') -> Dict:
    """
    Executa ANOVA two-way (simplificada usando scipy).
    
    Nota: Esta é uma implementação simplificada. Para análise completa,
    considere usar statsmodels ou R.
    
    Args:
        data: DataFrame com dados
        factor1: Nome da primeira coluna com fator
        factor2: Nome da segunda coluna com fator
        value_col: Nome da coluna com valores
        
    Returns:
        Dicionário com resultados da ANOVA
    """
    # Para ANOVA two-way completa, seria necessário usar statsmodels
    # Por enquanto, retornamos uma estrutura básica
    return {
        'note': 'ANOVA two-way completa requer statsmodels.anova_lm ou R',
        'factor1': factor1,
        'factor2': factor2,
        'suggestion': 'Use statsmodels.stats.anova.anova_lm para análise completa'
    }


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


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Análise estatística: ANOVA + Tukey HSD'
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
        help='Diretório de saída para resultados'
    )
    
    args = parser.parse_args()
    
    # Cria diretórios de saída
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'tables').mkdir(exist_ok=True)
    
    # Carrega dados
    print(f"Carregando dados de: {args.csv_path}")
    df = load_results_csv(args.csv_path)
    print(f"Total de linhas: {len(df)}")
    
    # Filtra dados
    print("\nFiltrando dados (split=test, epoch_type=best, metric=accuracy)...")
    filtered = filter_data_for_analysis(df)
    print(f"Linhas após filtro: {len(filtered)}")
    
    if len(filtered) == 0:
        print("ERRO: Nenhum dado encontrado após filtros!")
        return
    
    # Fixa ordem de test_quality
    filtered = fix_test_quality_order(filtered)
    
    # Análises principais
    print("\n" + "="*60)
    print("ANÁLISE ESTATÍSTICA")
    print("="*60)
    
    # Para cada modelo e train_quality
    models = filtered['model'].unique()
    train_qualities = filtered['train_quality'].unique()
    
    all_anova_results = []
    all_tukey_results = []
    all_assumptions = []
    
    for model in models:
        for train_q in train_qualities:
            print(f"\n--- Modelo: {model}, Train Quality: {train_q} ---")
            
            # Filtra dados para esta combinação
            subset = filtered[
                (filtered['model'] == model) &
                (filtered['train_quality'] == train_q)
            ].copy()
            
            if len(subset) == 0:
                continue
            
            # Verifica suposições
            print("Verificando suposições...")
            assumptions = check_assumptions(subset, 'test_quality')
            assumptions['model'] = model
            assumptions['train_quality'] = train_q
            all_assumptions.append(assumptions)
            
            print(f"  Homocedasticidade (Levene): p={assumptions['homogeneity'].get('p_value', 'N/A'):.4f}")
            
            # ANOVA one-way
            print("Executando ANOVA one-way...")
            anova_result = one_way_anova(subset, 'test_quality')
            if 'error' not in anova_result:
                anova_result['model'] = model
                anova_result['train_quality'] = train_q
                all_anova_results.append(anova_result)
                
                print(f"  F-statistic: {anova_result['f_statistic']:.4f}")
                print(f"  p-value: {anova_result['p_value']:.6f}")
                print(f"  Significativo: {anova_result['significant']}")
            
            # Pós-teste
            homogeneous = assumptions['homogeneity'].get('homogeneous', True)
            
            if homogeneous:
                print("Executando Tukey HSD (variâncias homogêneas)...")
                tukey_result = tukey_hsd(subset, 'test_quality')
                if len(tukey_result) > 0:
                    tukey_result['model'] = model
                    tukey_result['train_quality'] = train_q
                    all_tukey_results.append(tukey_result)
                    print(f"  {len(tukey_result)} comparações realizadas")
            else:
                print("Executando Games-Howell (variâncias heterogêneas)...")
                games_result = games_howell(subset, 'test_quality')
                if len(games_result) > 0:
                    games_result['model'] = model
                    games_result['train_quality'] = train_q
                    games_result['test'] = 'Games-Howell'
                    all_tukey_results.append(games_result)
                    print(f"  {len(games_result)} comparações realizadas")
    
    # Salva resultados
    print("\n" + "="*60)
    print("SALVANDO RESULTADOS")
    print("="*60)
    
    # ANOVA results
    if all_anova_results:
        anova_df = pd.DataFrame(all_anova_results)
        anova_path = output_dir / 'tables' / 'anova_results.csv'
        anova_df.to_csv(anova_path, index=False)
        print(f"Resultados ANOVA salvos em: {anova_path}")
    
    # Tukey/Games-Howell results
    if all_tukey_results:
        tukey_df = pd.concat(all_tukey_results, ignore_index=True)
        tukey_path = output_dir / 'tables' / 'tukey_results.csv'
        tukey_df.to_csv(tukey_path, index=False)
        print(f"Resultados Tukey/Games-Howell salvos em: {tukey_path}")
    
    # Assumptions
    if all_assumptions:
        assumptions_path = output_dir / 'tables' / 'assumptions_check.json'
        with open(assumptions_path, 'w') as f:
            json.dump(all_assumptions, f, indent=2)
        print(f"Verificação de suposições salva em: {assumptions_path}")
    
    print("\nAnálise estatística concluída!")


if __name__ == '__main__':
    main()
