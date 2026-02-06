# ROADMAP - LIMLA

Detailed roadmap of the LIMLA project, aligned with the [subproject.md](subproject.md) document. Completed items include the commit date that implemented them. Pending items have no date.

**Legend:**
- `[x]` = Completed
- `[ ]` = Pending

---

## Stage 1: Survey and Selection of Media Types

> Systematic survey of publicly recognized databases as ML benchmarks, covering images, audio, and video. Criteria: diversity of scenarios, availability of annotations, representativeness of real-world conditions.

### Images

- [x] Generic structure for image datasets (folders per class) — 2025-11-16
- [ ] Document and integrate benchmark databases for classification (e.g., ImageNet, CIFAR-10/100)
- [ ] Document and integrate benchmark databases for detection (e.g., COCO, Pascal VOC)

### Audio

- [ ] Survey of audio databases for ML
- [ ] Selection and documentation of chosen databases
- [ ] Integration of databases into the pipeline

### Video

- [ ] Survey of video databases for ML
- [ ] Selection and documentation of chosen databases
- [ ] Integration of databases into the pipeline

---

## Stage 2: Survey and Application of Lossy Compression Algorithms

> Apply multiple compression levels, varying QP and QF.

### Images

- [x] JPEG compression with configurable QF (1–100) — 2025-09-04
- [x] Support for .jpg and .jpeg images — 2026-01-08
- [x] Multiple quality levels per run — 2025-12-14
- [ ] JPEG2000 compression with quality levels
- [ ] QP parameter for algorithms that use it

### Video

- [ ] H.264 compression with configurable QP
- [ ] HEVC compression with configurable QP

### Audio

- [ ] MP3 compression with quality levels
- [ ] Opus compression with quality levels

### Pipeline

- [x] Parallel processing (rayon) — 2025-12-16
- [x] Progress bar — 2025-12-16
- [x] Configurable train/val/test split — 2025-12-14
- [x] Hierarchical organization by quality (q1, q5, q10...) — 2025-12-17

---

## Stage 3: Selection and Training of Deep Neural Network Architectures

> Train with both original and compressed data to evaluate response to compression during training and inference.

### Image classification

- [x] MobileNetV2 CNN for classification — 2025-12-16
- [x] VGG16 CNN for classification — 2025-12-17
- [x] Transfer learning with pre-trained models — 2025-12-16
- [x] Configurable early stopping — 2025-12-17
- [x] CLI for training (model, epochs, batch_size, etc.) — 2025-12-17
- [ ] Option to train with original (uncompressed) data for baseline
- [ ] Additional classification architectures (optional)

### Object detection (images and video)

- [ ] Detection model for images (e.g., YOLO, Faster R-CNN)
- [ ] Detection model for video
- [ ] Training and evaluation pipeline for detection

### Audio

- [ ] Architecture for audio processing
- [ ] Training pipeline for audio

### Training infrastructure

- [x] Rust → CNN integration (`run` command after compression) — 2025-12-17
- [x] Script for individual image testing — 2025-12-17
- [x] Checkpoint saving (best.pt, last.pt) — 2025-12-17
- [x] Class mapping (classes.json) — 2025-12-17

---

## Stage 4: Quantitative Evaluation and Robustness Analysis

> Metrics for classification/detection and for adversarial robustness.

### Classification and detection metrics

- [x] Accuracy — 2025-12-16
- [x] Loss (training and validation) — 2025-12-16
- [x] CSV export of results — 2026-01-08
- [ ] Precision
- [ ] Recall
- [ ] F1-score
- [ ] Detection-specific metrics (mAP, IoU, etc.)

### Adversarial robustness metrics

- [ ] Robust Accuracy
- [ ] TASR (Test-time Adversarial Success Rate or similar)
- [ ] Evaluation at different compression levels

---

## Objective 4: Adversarial Attacks

> Analyze the interaction between compression and adversarial attacks (mitigation vs. vulnerabilities).

- [ ] Implement adversarial attacks (e.g., FGSM, PGD)
- [ ] Pipeline to generate adversarial examples
- [ ] Evaluate models under attacks on compressed vs. original data
- [ ] Analysis: compression as defense or amplifier of vulnerabilities
- [ ] Documentation of results

---

## Infrastructure and Documentation

- [x] Project initialization — 2025-09-04
- [x] Unit tests for JPEG compression — 2025-09-04
- [x] CI/CD (GitHub Actions) — 2025-09-04
- [x] CLI with clap (Rust) — 2025-12-17
- [x] subproject.md document in repository — 2025-12-17
- [x] Project README — 2025-12-17
- [x] CNN module README — 2025-12-17
- [ ] Documentation of used databases
- [ ] Experiment reproduction guide

---

## Summary by status

| Category           | Completed | Pending |
|--------------------|-----------|---------|
| Stage 1 (Media)    | 1         | 6       |
| Stage 2 (Compression) | 8      | 7       |
| Stage 3 (Models)   | 10        | 7       |
| Stage 4 (Metrics)  | 3         | 6       |
| Adversarial attacks | 0        | 5       |
| Infrastructure     | 7         | 2       |

---

## Suggested implementation order

1. **Classification metrics** — Precision, Recall, F1-score (complements existing)
2. **Training with original data** — Baseline for comparison
3. **JPEG2000** — Second compression algorithm for images
4. **Benchmark databases** — Document and integrate recognized datasets
5. **Detection models** — YOLO or similar for images
6. **Adversarial attacks** — Implementation and analysis
7. **Audio** — Full pipeline (compression + models)
8. **Video** — Full pipeline (compression + models)
