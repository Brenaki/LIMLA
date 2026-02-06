## ANALYZING THE IMPACT OF LOSSY COMPRESSION ON MEDIA IN MACHINE LEARNING ALGORITHMS

### 1. KEYWORDS

`Lossy compression` • `Machine learning` • `Media quality` • `Neural networks` • `Adversarial attack` • `Model robustness`

---

### 2. ABSTRACT

The growing use of Machine Learning (ML), particularly Deep Learning (DL), to process digital media such as images, audio, and video is fundamental in areas such as healthcare and security. The massive volume of data requires lossy compression to optimize storage and transmission, especially in resource-constrained environments.

However, compression introduces artifacts and degradations that can significantly compromise ML model performance in tasks such as classification, detection, and recognition. Examples include the drop in accuracy in facial recognition under high compression and the degradation in object detection.

The complex relationship between compression and adversarial robustness is also relevant, as compression can both mitigate and amplify attacks. This study systematically investigates the impact of compression on different media and neural networks to understand how degradations affect performance and robustness.

The results aim to inform the construction of more reliable ML systems and provide practical recommendations for the use of compressed data in real-world applications.

---

### 3. CHARACTERIZATION AND JUSTIFICATION

The growing use of Machine Learning (ML) algorithms, especially Deep Learning (DL), to process digital media such as images, audio, and video is vital in many areas, including healthcare, security, and automotive (Ferreira; Do Couto; De Melo Baptista Domingues, 2025; Jacobellis; Cummings; Yadwadkar, 2024).

However, the large volume of this data often requires the use of **lossy compression** to optimize storage and transmission, particularly in locations with limited resources or bandwidth constraints (Baris et al., 2025; Ferreira; Do Couto; De Melo Baptista Domingues, 2025; Yasin; Abdulazeez, 2021).

#### Impacts of Lossy Compression

Lossy compression, although efficient in reducing file sizes, introduces artifacts and degradations that can severely compromise ML model performance:

- **Facial recognition:** Accuracy may decrease under severe compression (Ferreira; Do Couto; De Melo Baptista Domingues, 2025)
- **Object detection:** Performance suffers degradation, especially with additional noise (Baris et al., 2025)
- **Medical images:** Even with common irreversible compression (JPEG, JPEG2000), the impact on DL models varies, and there is no consensus on the compression threshold (Urbaniak, 2024)
- **Audio:** Models may have reduced performance when encountering encoding parameters not seen during training (Koops; Micchi; Quinton, 2024)

Additionally, compression can both mitigate and amplify adversarial attacks, which adds another layer of complexity to its relationship with ML model robustness (Niu; Yang, 2023; Yin et al., 2020).

#### Research Relevance

Given this, a systematic investigation of the impact of lossy compression on diverse media and neural network architectures is crucial (Ferreira; Do Couto; De Melo Baptista Domingues, 2025). The results can help build more robust, efficient, and secure ML systems and offer practical recommendations for the use of compressed media in real-world applications.

---

### 4. OBJECTIVES

#### General Objective

Systematically investigate the impact of lossy compression on digital media on ML model performance, considering compression severity, different types of media, neural network architectures, and their interaction in scenarios with adversarial attacks.

#### Specific Objectives

1. Evaluate the performance of DL models in specific tasks, such as classification and detection, using media subjected to different levels of compression severity

2. Compare the impact of compression degradations on different types of media, such as images, audio, and video, in ML models

3. Investigate the effect of lossy compression on deep neural networks during training and inference phases

4. Analyze the interaction between lossy compression and adversarial attacks, investigating whether compression can act as a defense or introduce additional vulnerabilities for ML models

---

### 5. METHODOLOGY AND ACTION STRATEGY

#### Stage 1: Survey and Selection of Media Types

Initially, a systematic survey of publicly recognized databases as ML benchmarks will be conducted, covering images, audio, and video.

The selection of these databases will consider:
- Diversity of scenarios and perception tasks (classification and detection)
- Availability of annotations
- Representativeness of real-world conditions, including presence of noise and artifacts

This stage is fundamental to ensure that experiments reflect practical challenges faced in real applications (Baris et al., 2025; Ferreira; Do Couto; De Melo Baptista Domingues, 2025; Jacobellis; Cummings; Yadwadkar, 2024; Niu; Yang, 2023; Obaid; Kadhim, 2025).

#### Stage 2: Survey and Application of Lossy Compression Algorithms

With the selected media, a survey of the main lossy compression algorithms relevant to each data type will be conducted:

- **Images and video:** JPEG, JPEG2000, H.264, HEVC
- **Audio:** MP3, Opus

Multiple compression levels will be applied, varying parameters such as:
- **QP** (Quantization Parameter)
- **QF** (Quality Factor)

This approach will allow analyzing the progressive impact of compression on data quality and, consequently, on model performance (Baris et al., 2025; Ferreira; Do Couto; De Melo Baptista Domingues, 2025; Obaid; Kadhim, 2025; Perepelytsia; Dellwo, 2023; Yin et al., 2020).

#### Stage 3: Selection and Training of Deep Neural Network Architectures

Next, deep neural network architectures suitable for each media type and task will be selected:

- **CNNs** for image classification
- **Detection models** for images and video
- **Specific architectures** for audio processing

Models will be trained with both original and compressed data, allowing evaluation of how different architectures respond to compression during training and inference (Ahmad et al., 2021; Baris et al., 2025; Ferreira; Do Couto; De Melo Baptista Domingues, 2025; Obaid; Kadhim, 2025; Urbaniak, 2024).

#### Stage 4: Quantitative Evaluation and Robustness Analysis

Model performance will be evaluated through appropriate quantitative metrics:

**For classification and detection:**
- Accuracy
- Precision
- Recall
- F1-score

**For adversarial robustness:**
- Robust Accuracy
- TASR

The evaluation will be conducted at different compression levels, seeking to understand the generalization capacity and robustness of models in realistic data degradation scenarios (Ahmad et al., 2021; Baris et al., 2025; Ferreira; Couto, 2025; Ferreira; Do Couto; De Melo Baptista Domingues, 2025; Niu; Yang, 2023).

---

### 6. EXPECTED RESULTS

It is expected to identify how different levels of lossy compression affect the performance of ML algorithms in classification and detection tasks, considering different types of media and neural network architectures.

The results should:

✓ Indicate acceptable compression limits for each media type and application

✓ Provide support for the development of models more robust to compression artifacts

✓ Contribute to understanding the relationship between compression and adversarial attacks

---

### 7. REFERENCES

AHMAD, Aanis et al. Performance of deep learning models for classifying and detecting common weeds in corn and soybean production systems. **Computers and Electronics in Agriculture**, v. 184, p. 106081, May 2021.

BARIS, Gabriele et al. Automotive DNN-Based Object Detection in the Presence of Lens Obstruction and Video Compression. **IEEE Access**, v. 13, p. 36575-36589, 2025.

FERREIRA, Fernando Rodrigues Trindade; COUTO, Loena Marins Do. Development of a computational deep learning model for detecting people in aerial images and videos degraded by compression artifacts. **Earth Science Informatics**, v. 18, n. 2, p. 408, Jun. 2025.

FERREIRA, Fernando Rodrigues Trindade; DO COUTO, Loena Marins; DE MELO BAPTISTA DOMINGUES, Guilherme. Comparing the efficiency of YOLO-M for face recognition in images and videos degraded by compression artifacts. **Evolving Systems**, v. 16, n. 2, p. 70, Jun. 2025.

JACOBELLIS, Dan; CUMMINGS, Daniel; YADWADKAR, Neeraja J. **Machine Perceptual Quality: Evaluating the Impact of Severe Lossy Compression on Audio and Image Models**. arXiv, Jan. 15, 2024. Available at: <http://arxiv.org/abs/2401.07957>. Accessed: Jun. 8, 2025.

KOOPS, Hendrik Vincent; MICCHI, Gianluca; QUINTON, Elio. **Robust Lossy Audio Compression Identification**. arXiv, Jul. 31, 2024. Available at: <http://arxiv.org/abs/2407.21545>. Accessed: Jun. 8, 2025.

NIU, Zhong-Han; YANG, Yu-Bin. Defense Against Adversarial Attacks with Efficient Frequency-Adaptive Compression and Reconstruction. **Pattern Recognition**, v. 138, p. 109382, Jun. 2023.

OBAID, Ali A.; KADHIM, Hasan M. Deep Learning for Lossless Audio Compression. **Journal of Engineering**, v. 31, n. 4, p. 100-112, Apr. 1, 2025.

PEREPELYTSIA, Valeriia; DELLWO, Volker. Acoustic compression in Zoom audio does not compromise voice recognition performance. **Scientific Reports**, v. 13, n. 1, p. 18742, Oct. 31, 2023.

URBANIAK, Ilona Anna. Using Compressed JPEG and JPEG2000 Medical Images in Deep Learning: A Review. **Applied Sciences**, v. 14, n. 22, p. 10524, Nov. 15, 2024.

YASIN, Hajar Maseeh; ABDULAZEEZ, Adnan Mohsin. Image Compression Based on Deep Learning: A Review. **Asian Journal of Research in Computer Science**, p. 62-76, May 1, 2021.

YIN, Zhaoxia et al. Defense against adversarial attacks by low‐level image transformations. **International Journal of Intelligent Systems**, v. 35, n. 10, p. 1453-1466, Oct. 2020.

---

### 8. EXECUTION SCHEDULE

| **Activities** | **Sep/25** | **Oct** | **Nov** | **Dec** | **Jan** | **Feb** | **Mar** | **Apr** | **May** | **Jun** | **Jul** | **Aug/26** |
|----------------|:----------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:----------:|
| Database selection and media compression | ✓ | ✓ | | | | | | | | | | |
| Model training and evaluation | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | | |
| **Semester Report** (by last business day of March 2026) | | | | | | | ✓ | | | | | |
| Adversarial attack implementation and analysis | | | | | | | ✓ | ✓ | ✓ | ✓ | | |
| Results analysis and recommendation development | | | | | | | | | ✓ | ✓ | ✓ | |
| **Final Report** (by last business day of September 2026) | | | | | | | | | | | | ✓ |
| EAIC 2026 registration and paper submission | | | | | | | | | | | | TBD |

---
