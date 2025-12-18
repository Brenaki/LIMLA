## ANALISAR O IMPACTO DA COMPRESSÃO COM PERDAS EM MÍDIAS NOS ALGORITMOS DE MACHINE LEARNING

### 1. PALAVRAS-CHAVE

`Compressão com perdas` • `Machine learning` • `Qualidade de mídia` • `Redes neurais` • `Adversarial attack` • `Robustez de modelos`

---

### 2. RESUMO

O uso crescente de Machine Learning (ML), em particular Deep Learning (DL), para processar mídias digitais como imagens, áudio e vídeo é fundamental em áreas como saúde e segurança. O volume massivo de dados requer compressão com perdas para otimizar armazenamento e transmissão, especialmente em ambientes com recursos limitados. 

Contudo, a compressão introduz artefatos e degradações que podem comprometer significativamente o desempenho de modelos ML em tarefas como classificação, detecção e reconhecimento. Exemplos incluem a queda na acurácia em reconhecimento facial sob alta compressão e a degradação na detecção de objetos. 

A relação complexa entre compressão e robustez adversarial também é relevante, pois a compressão pode tanto mitigar quanto potencializar ataques. Este estudo investiga sistematicamente o impacto da compressão em diferentes mídias e redes neurais para entender como as degradações afetam o desempenho e a robustez. 

Os resultados visam informar a construção de sistemas ML mais confiáveis e fornecer recomendações práticas para o uso de dados comprimidos em aplicações reais.

---

### 3. CARACTERIZAÇÃO E JUSTIFICATIVA

O uso crescente de algoritmos de Machine Learning (ML), especialmente Deep Learning (DL), para processar mídias digitais como imagens, áudios e vídeos é vital em muitas áreas, como saúde, segurança e automotiva (Ferreira; Do Couto; De Melo Baptista Domingues, 2025; Jacobellis; Cummings; Yadwadkar, 2024). 

Contudo, o grande volume desses dados exige frequentemente o uso de **compressão com perdas** (*lossy compression*) para otimizar armazenamento e transmissão, particularmente em locais com recursos limitados ou restrições de banda (Baris et al., 2025; Ferreira; Do Couto; De Melo Baptista Domingues, 2025; Yasin; Abdulazeez, 2021).

#### Impactos da Compressão com Perdas

A compressão com perdas, embora eficiente em reduzir o tamanho dos arquivos, introduz artefatos e degradações que podem comprometer severamente o desempenho dos modelos de ML:

- **Reconhecimento facial:** A acurácia pode diminuir sob compressão severa (Ferreira; Do Couto; De Melo Baptista Domingues, 2025)
- **Detecção de objetos:** O desempenho sofre degradação, especialmente com ruído adicional (Baris et al., 2025)
- **Imagens médicas:** Mesmo com compressão irreversível comum (JPEG, JPEG2000), o impacto nos modelos DL varia, e não há consenso sobre o limiar de compressão (Urbaniak, 2024)
- **Áudio:** Modelos podem ter desempenho reduzido ao encontrar parâmetros de codificação não vistos no treinamento (Koops; Micchi; Quinton, 2024)

Adicionalmente, a compressão pode tanto mitigar quanto potencializar ataques adversariais, o que adiciona outra camada de complexidade à sua relação com a robustez de modelos ML (Niu; Yang, 2023; Yin et al., 2020).

#### Relevância da Pesquisa

Diante disso, uma investigação sistemática do impacto da compressão com perdas em diversas mídias e arquiteturas de redes neurais é crucial (Ferreira; Do Couto; De Melo Baptista Domingues, 2025). Os resultados podem ajudar a construir sistemas de ML mais robustos, eficientes e seguros e oferecer recomendações práticas para o uso de mídias comprimidas em aplicações do mundo real.

---

### 4. OBJETIVOS

#### Objetivo Geral

Investigar sistematicamente o impacto da compressão com perdas em mídias digitais no desempenho de modelos de ML, considerando a severidade da compressão, diferentes tipos de mídia, arquiteturas de redes neurais e a sua interação em cenários com ataques adversariais.

#### Objetivos Específicos

1. Avaliar o desempenho de modelos de DL em tarefas específicas, como classificação e detecção, utilizando mídias submetidas a diferentes níveis de severidade de compressão

2. Comparar o impacto das degradações da compressão em diferentes tipos de mídia, como imagens, áudios e vídeos, em modelos de ML

3. Investigar o efeito da compressão com perdas em redes neurais profundas durante as fases de treinamento e inferência

4. Analisar a interação entre compressão com perdas e ataques adversariais, investigando se a compressão pode atuar como defesa ou introduzir vulnerabilidades adicionais para os modelos de ML

---

### 5. METODOLOGIA E ESTRATÉGIA DE AÇÃO

#### Etapa 1: Levantamento e Seleção de Tipos de Mídia

Inicialmente, será realizado um levantamento sistemático de bases de dados públicas amplamente reconhecidas como benchmarks em ML, abrangendo imagens, áudios e vídeos. 

A seleção dessas bases considerará:
- Diversidade de cenários e tarefas de percepção (classificação e detecção)
- Disponibilidade de anotações
- Representatividade de condições reais, incluindo presença de ruído e artefatos

Esta etapa é fundamental para garantir que os experimentos reflitam desafios práticos enfrentados em aplicações reais (Baris et al., 2025; Ferreira; Do Couto; De Melo Baptista Domingues, 2025; Jacobellis; Cummings; Yadwadkar, 2024; Niu; Yang, 2023; Obaid; Kadhim, 2025).

#### Etapa 2: Levantamento e Aplicação de Algoritmos de Compressão com Perda

Com as mídias selecionadas, será feito um levantamento dos principais algoritmos de compressão com perda relevantes para cada tipo de dado:

- **Imagens e vídeos:** JPEG, JPEG2000, H.264, HEVC
- **Áudio:** MP3, Opus

Serão aplicados múltiplos níveis de compressão, variando parâmetros como:
- **QP** (Quantization Parameter)
- **QF** (Quality Factor)

Essa abordagem permitirá analisar o impacto progressivo da compressão sobre a qualidade dos dados e, consequentemente, sobre o desempenho dos modelos (Baris et al., 2025; Ferreira; Do Couto; De Melo Baptista Domingues, 2025; Obaid; Kadhim, 2025; Perepelytsia; Dellwo, 2023; Yin et al., 2020).

#### Etapa 3: Seleção e Treinamento de Arquiteturas de Redes Neurais Profundas

Na sequência, serão selecionadas arquiteturas de redes neurais profundas adequadas a cada mídia e tarefa:

- **CNNs** para classificação de imagens
- **Modelos de detecção** para imagens e vídeos
- **Arquiteturas específicas** para processamento de áudio

Os modelos serão treinados tanto com dados originais quanto com versões comprimidas, permitindo avaliar como diferentes arquiteturas respondem à compressão durante o treinamento e a inferência (Ahmad et al., 2021; Baris et al., 2025; Ferreira; Do Couto; De Melo Baptista Domingues, 2025; Obaid; Kadhim, 2025; Urbaniak, 2024).

#### Etapa 4: Avaliação Quantitativa e Análise de Robustez

O desempenho dos modelos será avaliado por meio de métricas quantitativas apropriadas:

**Para classificação e detecção:**
- Acurácia
- Precisão
- Recall
- F1-score

**Para robustez adversarial:**
- Robust Accuracy
- TASR

A avaliação será conduzida em diferentes níveis de compressão, buscando compreender a capacidade de generalização e a robustez dos modelos em cenários realistas de degradação dos dados (Ahmad et al., 2021; Baris et al., 2025; Ferreira; Couto, 2025; Ferreira; Do Couto; De Melo Baptista Domingues, 2025; Niu; Yang, 2023).

---

### 6. RESULTADOS ESPERADOS

Espera-se identificar como diferentes níveis de compressão com perdas afetam o desempenho de algoritmos de ML em tarefas de classificação e detecção, considerando diferentes tipos de mídia e arquiteturas de redes neurais. 

Os resultados devem:

✓ Indicar limites de compressão aceitáveis para cada tipo de mídia e aplicação

✓ Fornecer subsídios para o desenvolvimento de modelos mais robustos a artefatos de compressão

✓ Contribuir para a compreensão da relação entre compressão e ataques adversariais

---

### 7. REFERÊNCIAS

AHMAD, Aanis et al. Performance of deep learning models for classifying and detecting common weeds in corn and soybean production systems. **Computers and Electronics in Agriculture**, v. 184, p. 106081, maio 2021.

BARIS, Gabriele et al. Automotive DNN-Based Object Detection in the Presence of Lens Obstruction and Video Compression. **IEEE Access**, v. 13, p. 36575-36589, 2025.

FERREIRA, Fernando Rodrigues Trindade; COUTO, Loena Marins Do. Development of a computational deep learning model for detecting people in aerial images and videos degraded by compression artifacts. **Earth Science Informatics**, v. 18, n. 2, p. 408, jun. 2025.

FERREIRA, Fernando Rodrigues Trindade; DO COUTO, Loena Marins; DE MELO BAPTISTA DOMINGUES, Guilherme. Comparing the efficiency of YOLO-M for face recognition in images and videos degraded by compression artifacts. **Evolving Systems**, v. 16, n. 2, p. 70, jun. 2025.

JACOBELLIS, Dan; CUMMINGS, Daniel; YADWADKAR, Neeraja J. **Machine Perceptual Quality: Evaluating the Impact of Severe Lossy Compression on Audio and Image Models**. arXiv, 15 jan. 2024. Disponível em: <http://arxiv.org/abs/2401.07957>. Acesso em: 8 jun. 2025.

KOOPS, Hendrik Vincent; MICCHI, Gianluca; QUINTON, Elio. **Robust Lossy Audio Compression Identification**. arXiv, 31 jul. 2024. Disponível em: <http://arxiv.org/abs/2407.21545>. Acesso em: 8 jun. 2025.

NIU, Zhong-Han; YANG, Yu-Bin. Defense Against Adversarial Attacks with Efficient Frequency-Adaptive Compression and Reconstruction. **Pattern Recognition**, v. 138, p. 109382, jun. 2023.

OBAID, Ali A.; KADHIM, Hasan M. Deep Learning for Lossless Audio Compression. **Journal of Engineering**, v. 31, n. 4, p. 100-112, 1 abr. 2025.

PEREPELYTSIA, Valeriia; DELLWO, Volker. Acoustic compression in Zoom audio does not compromise voice recognition performance. **Scientific Reports**, v. 13, n. 1, p. 18742, 31 out. 2023.

URBANIAK, Ilona Anna. Using Compressed JPEG and JPEG2000 Medical Images in Deep Learning: A Review. **Applied Sciences**, v. 14, n. 22, p. 10524, 15 nov. 2024.

YASIN, Hajar Maseeh; ABDULAZEEZ, Adnan Mohsin. Image Compression Based on Deep Learning: A Review. **Asian Journal of Research in Computer Science**, p. 62-76, 1 maio 2021.

YIN, Zhaoxia et al. Defense against adversarial attacks by low‐level image transformations. **International Journal of Intelligent Systems**, v. 35, n. 10, p. 1453-1466, out. 2020.

---

### 8. CRONOGRAMA DE EXECUÇÃO

| **Atividades** | **Set/25** | **Out** | **Nov** | **Dez** | **Jan** | **Fev** | **Mar** | **Abr** | **Mai** | **Jun** | **Jul** | **Ago/26** |
|----------------|:----------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:----------:|
| Seleção de bases de dados e compressão das mídias | ✓ | ✓ | | | | | | | | | | |
| Treinamento e avaliação dos modelos | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | | |
| **Relatório Semestral** (até último dia útil de março/2026) | | | | | | | ✓ | | | | | |
| Implementação de ataques adversariais e análise | | | | | | | ✓ | ✓ | ✓ | ✓ | | |
| Análise dos resultados e elaboração de recomendações | | | | | | | | | ✓ | ✓ | ✓ | |
| **Relatório Final** (até último dia útil de setembro/2026) | | | | | | | | | | | | ✓ |
| Inscrição e submissão de trabalho no EAIC 2026 | | | | | | | | | | | | A definir |

---