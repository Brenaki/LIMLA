# ROADMAP - LIMLA

Roadmap detalhado do projeto LIMLA, alinhado ao documento [subproject.md](./subproject.md). Itens concluídos incluem data do commit que os implementou. Itens pendentes não possuem data.

**Legenda:**
- `[x]` = Concluído
- `[ ]` = Pendente

---

## Etapa 1: Levantamento e Seleção de Tipos de Mídia

> Levantamento sistemático de bases de dados públicas reconhecidas como benchmarks em ML, abrangendo imagens, áudios e vídeos. Critérios: diversidade de cenários, disponibilidade de anotações, representatividade de condições reais.

### Imagens

- [x] Estrutura genérica para datasets de imagens (pastas por classe) — 2025-11-16
- [ ] Documentar e integrar bases benchmark para classificação (ex.: ImageNet, CIFAR-10/100)
- [ ] Documentar e integrar bases benchmark para detecção (ex.: COCO, Pascal VOC)

### Áudio

- [ ] Levantamento de bases de áudio para ML
- [ ] Seleção e documentação das bases escolhidas
- [ ] Integração das bases no pipeline

### Vídeo

- [ ] Levantamento de bases de vídeo para ML
- [ ] Seleção e documentação das bases escolhidas
- [ ] Integração das bases no pipeline

---

## Etapa 2: Levantamento e Aplicação de Algoritmos de Compressão com Perda

> Aplicar múltiplos níveis de compressão, variando QP e QF.

### Imagens

- [x] Compressão JPEG com QF configurável (1–100) — 2025-09-04
- [x] Suporte a imagens .jpg e .jpeg — 2026-01-08
- [x] Múltiplos níveis de qualidade por execução — 2025-12-14
- [ ] Compressão JPEG2000 com níveis de qualidade
- [ ] Parâmetro QP para algoritmos que o utilizam

### Vídeo

- [ ] Compressão H.264 com QP configurável
- [ ] Compressão HEVC com QP configurável

### Áudio

- [ ] Compressão MP3 com níveis de qualidade
- [ ] Compressão Opus com níveis de qualidade

### Pipeline

- [x] Processamento paralelo (rayon) — 2025-12-16
- [x] Barra de progresso — 2025-12-16
- [x] Split train/val/test configurável — 2025-12-14
- [x] Organização hierárquica por qualidade (q1, q5, q10...) — 2025-12-17

---

## Etapa 3: Seleção e Treinamento de Arquiteturas de Redes Neurais Profundas

> Treinar com dados originais e comprimidos para avaliar resposta à compressão em treino e inferência.

### Classificação de imagens

- [x] CNN MobileNetV2 para classificação — 2025-12-16
- [x] CNN VGG16 para classificação — 2025-12-17
- [x] Transfer learning com modelos pré-treinados — 2025-12-16
- [x] Early stopping configurável — 2025-12-17
- [x] CLI para treinamento (modelo, epochs, batch_size, etc.) — 2025-12-17
- [ ] Opção de treinar com dados originais (sem compressão) para baseline
- [ ] Outras arquiteturas de classificação (opcional)

### Detecção de objetos (imagens e vídeo)

- [ ] Modelo de detecção para imagens (ex.: YOLO, Faster R-CNN)
- [ ] Modelo de detecção para vídeo
- [ ] Pipeline de treinamento e avaliação para detecção

### Áudio

- [ ] Arquitetura para processamento de áudio
- [ ] Pipeline de treinamento para áudio

### Infraestrutura de treinamento

- [x] Integração Rust → CNN (comando `run` após compressão) — 2025-12-17
- [x] Script para teste de imagem individual — 2025-12-17
- [x] Salvamento de checkpoints (best.pt, last.pt) — 2025-12-17
- [x] Mapeamento de classes (classes.json) — 2025-12-17

---

## Etapa 4: Avaliação Quantitativa e Análise de Robustez

> Métricas para classificação/detecção e para robustez adversarial.

### Métricas de classificação e detecção

- [x] Acurácia — 2025-12-16
- [x] Loss (treino e validação) — 2025-12-16
- [x] Exportação de resultados em CSV — 2026-01-08
- [ ] Precisão (Precision)
- [ ] Recall
- [ ] F1-score
- [ ] Métricas específicas para detecção (mAP, IoU, etc.)

### Métricas de robustez adversarial

- [ ] Robust Accuracy
- [ ] TASR (Test-time Adversarial Success Rate ou similar)
- [ ] Avaliação em diferentes níveis de compressão

---

## Objetivo 4: Ataques Adversariais

> Analisar interação entre compressão e ataques adversariais (mitigação vs. vulnerabilidades).

- [ ] Implementar ataques adversariais (ex.: FGSM, PGD)
- [ ] Pipeline para gerar exemplos adversariais
- [ ] Avaliar modelos sob ataques em dados comprimidos vs. originais
- [ ] Análise: compressão como defesa ou amplificador de vulnerabilidades
- [ ] Documentação dos resultados

---

## Infraestrutura e Documentação

- [x] Inicialização do projeto — 2025-09-04
- [x] Testes unitários para compressão JPEG — 2025-09-04
- [x] CI/CD (GitHub Actions) — 2025-09-04
- [x] CLI com clap (Rust) — 2025-12-17
- [x] Documento subproject.md no repositório — 2025-12-17
- [x] README do projeto — 2025-12-17
- [x] README do módulo CNN — 2025-12-17
- [ ] Documentação das bases de dados utilizadas
- [ ] Guia de reprodução dos experimentos

---

## Resumo por status

| Categoria              | Concluído | Pendente |
|------------------------|-----------|----------|
| Etapa 1 (Mídias)      | 1         | 6        |
| Etapa 2 (Compressão)  | 8         | 7        |
| Etapa 3 (Modelos)     | 10        | 7        |
| Etapa 4 (Métricas)    | 3         | 6        |
| Ataques adversariais  | 0         | 5        |
| Infraestrutura        | 7         | 2        |

---

## Ordem sugerida de implementação

1. **Métricas de classificação** — Precision, Recall, F1-score (complementa o que já existe)
2. **Treino com dados originais** — Baseline para comparação
3. **JPEG2000** — Segundo algoritmo de compressão para imagens
4. **Bases benchmark** — Documentar e integrar datasets reconhecidos
5. **Modelos de detecção** — YOLO ou similar para imagens
6. **Ataques adversariais** — Implementação e análise
7. **Áudio** — Pipeline completo (compressão + modelos)
8. **Vídeo** — Pipeline completo (compressão + modelos)
