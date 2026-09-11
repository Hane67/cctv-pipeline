# Robust CCTV Video Analytics Platform

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()
[![Architecture](https://img.shields.io/badge/Architecture-Modular%20Pipeline-orange.svg)]()

Production-grade research and surveillance platform implementing the four problem statements from:
> **"Robust CCTV Video Analytics Under Illumination Changes and Compression Artifacts"**  
> *Course: CSEDS743E04, CHRIST (Deemed to be University)*

---

## System Architecture

```mermaid
flowchart TD
    subgraph Input["Surveillance Input Sources"]
        A[RTSP Stream / USB Webcam / Video File / Synthetic Generator]
    end

    subgraph S1["Stage 1: Adaptive Preprocessing"]
        B[Scene Luminance Analyzer] --> C{Day / Night / Glare}
        C -->|Dynamic Gamma + CLAHE| D[AdaDeband Selective Filter]
        D --> E[VRCNN Edge-Preserving Deblocking]
        E --> F[Adaptive Spatial-Temporal Denoising]
    end

    subgraph S2["Stage 2: Illumination-Invariant Segmentation"]
        G[Vectorized SuBSENSE with LBSP]
        H[Baselines: MOG2 / KNN]
        I[Morphological Cleanup & Shadow Suppression]
    end

    subgraph S3["Stage 3: Compression Impact & ROI Coding"]
        J[H.264 / H.265 Constant-QP Sweeper (18-51)]
        K[PSNR / SSIM / Bitrate Quantification]
        L[Operational Knee-Point Discovery]
        M[VCM / ROI Foreground-Guided Compression]
    end

    subgraph S4["Stage 4: Analytics, Detection & Tracking"]
        N[YOLOv8 Multi-Class Detector]
        O[ByteTrack / BoT-SORT Trajectory Tracker]
        P[Tripwire Line-Crossing & Counting]
        Q[Spatial Motion Heatmap Accumulator]
        R[CDRE Bitstream Distortion Metadata]
    end

    subgraph UI["Interfaces & Benchmarking"]
        S[Interactive Web Dashboard - Flask + MJPEG]
        T[Unified CLI Tool]
        U[CDNet2014 Metric Suite - F1, Precision, Recall, PWC]
    end

    A --> S1
    S1 --> S2
    S1 --> S4
    S2 -.-> M
    S4 --> UI
    S3 --> UI
    S2 --> U
```

---

## Research Problem Statements Mapping

| Problem Statement | Algorithmic Solution in Platform | Implementation Module |
|---|---|---|
| **1. Illumination & Artifacts** | Scene luminance classifier, dynamic CLAHE on LAB L-channel, auto-tuning gamma curve, AdaDeband gradient selective debanding, VRCNN bilateral deblocking, and SNR-adaptive night denoising. | [`cctv_pipeline.preprocessing`](file:///cctv_pipeline/preprocessing/) |
| **2. Illumination-Invariant Foreground** | Vectorized SuBSENSE: 8-bit relative-contrast Local Binary Similarity Patterns (LBSP), per-pixel dynamic distance threshold $R(x)$, adaptive update rate $T(x)$ suppressing mosquito noise, plus MOG2/KNN baselines and CDNet2014 F-measure evaluator. | [`cctv_pipeline.segmentation`](file:///cctv_pipeline/segmentation/) |
| **3. Compression Impact & VCM** | Constant-QP ladder sweep (QP 18 to 51), PSNR/SSIM/Bitrate quantification, detection retention curves, operational knee-point discovery ($>90\%$ retention boundary), and VCM foreground-guided ROI compression. | [`cctv_pipeline.compression`](file:///cctv_pipeline/compression/) |
| **4. End-to-End Analytics & Tracking** | YOLOv8 multi-class surveillance detector (person, vehicle, bicycle), ByteTrack trajectory tracking, virtual tripwire directional counting, motion density heatmaps, and CDRE side-data extractor. | [`cctv_pipeline.analytics`](file:///cctv_pipeline/analytics/) |

---

## Directory Structure

```
cctv_pipeline/
├── cctv_pipeline/                 # Core Python package
│   ├── config/                    # Pydantic v2 configuration schema & YAML defaults
│   │   ├── settings.py
│   │   └── default_config.yaml
│   ├── core/                      # Video I/O, logging, synthetic generator, visualizer
│   │   ├── video_stream.py
│   │   ├── visualizer.py
│   │   └── logger.py
│   ├── preprocessing/             # Stage 1: Illumination, debanding, deblocking, denoising
│   │   ├── illumination.py
│   │   ├── artifacts.py
│   │   ├── denoise.py
│   │   └── pipeline.py
│   ├── segmentation/              # Stage 2: SuBSENSE, baselines, CDNet2014 evaluation
│   │   ├── subsense.py
│   │   ├── baselines.py
│   │   └── metrics.py
│   ├── compression/               # Stage 3: QP sweep, knee point, ROI coding, PSNR/SSIM
│   │   ├── encoder.py
│   │   ├── sweep.py
│   │   ├── roi_coding.py
│   │   └── metrics.py
│   ├── analytics/                 # Stage 4: YOLOv8, ByteTrack, tripwires, heatmaps, CDRE
│   │   ├── detector.py
│   │   ├── tracker.py
│   │   ├── counting.py
│   │   ├── heatmap.py
│   │   └── cdre.py
│   ├── pipeline.py                # Unified Master Pipeline orchestrator
│   ├── cli.py                     # Command-line interface entrypoint
│   └── ui/                        # Modern interactive Web Dashboard
│       ├── web_app.py
│       └── templates/index.html
├── data/                          # Video storage and synthetic test generation
├── outputs/                       # Output videos, plots, and benchmark reports
├── tests/                         # Comprehensive unit and integration test suite
├── src/                           # Backward-compatibility shims for legacy scripts
├── pyproject.toml                 # Standard PEP 621 packaging metadata
├── requirements.txt               # Pinned production dependencies
└── README.md
```

---

## Installation & Setup

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Hane67/cctv-pipeline.git
cd cctv-pipeline

# Install required packages
pip install -r requirements.txt

# Or install as an editable package
pip install -e .
```

### 2. Native H.265 / HEVC Codec Support
- **Zero-Setup Native H.265**: Bundled `imageio-ffmpeg` guarantees a static FFmpeg binary with native `libx265` and hardware-accelerated HEVC encoders (`hevc_nvenc`, `hevc_qsv`, `hevc_amf`, `hevc_d3d12va`) without manual PATH configuration.
- **Automated Fallback**: If hardware encoders are absent, the platform transparently uses `libx265` software encoding or OpenCV discrete cosine transform (DCT) quantization simulation.

---

## Quickstart & Usage

### 1. Interactive Web Dashboard
Launch the modern web dashboard to view live multi-view video, adjust parameters in real-time, switch segmentation algorithms, review CDNet benchmarks, and trigger QP sweeps:
```bash
python -m cctv_pipeline ui --port 5000
```
Open **`http://localhost:5000`** in your browser.

**Dashboard Features:**
- **Live Stream View**: Quad-Split (all 4 stages), Side-by-Side (Input vs Output), Full Analytics HUD, or Foreground Mask.
- **Evaluation & Benchmark Suite**: Real-time H.265 compression ratio (e.g. 4.6x), bandwidth savings % (e.g. 78.4%), operational knee point QP, PSNR, SSIM, and CDNet 2014 metrics (Precision, Recall, F1, PWC).
- **Per-Stage Latency Distribution**: Stacked visualizer displaying real-time execution distribution across Preprocessing, Segmentation, and Tracking.
- **Click-to-Expand Graph Lightbox**: Click the sidebar QP curve thumbnail or table button to view and download full-resolution rate-distortion figures.

### 2. Run Pipeline on Video or Camera
Run the full 4-stage pipeline with a 4-way split screen (`quad`) showing:
- **Top-Left**: Raw Camera Input
- **Top-Right**: Stage 1 Adaptive Preprocessed
- **Bottom-Left**: Stage 2 Foreground Mask (SuBSENSE / MOG2 / KNN)
- **Bottom-Right**: Stage 4 Detection, Tracking, Tripwire Counters, and Telemetry HUD

```bash
# Run on a real CCTV pedestrian walkway clip
python -m cctv_pipeline run --source data/sample_cctv_people.mp4 --out outputs/annotated.mp4 --view quad

# Run on local webcam
python -m cctv_pipeline run --source 0 --view quad
```

Available `--view` modes:
- `quad`: 4-way split screen showcasing all research stages simultaneously.
- `side_by_side`: Original input vs. processed surveillance feed.
- `hud`: Full-resolution detection HUD with trajectory trails and tripwires.
- `mask_only`: High-contrast binary foreground segmentation mask.

### 3. Run Stage 3 QP Compression Sweep
Quantify the impact of H.264 / H.265 compression on machine vision accuracy:
```bash
python -m cctv_pipeline sweep --source data/synthetic_test.mp4 --qp-levels 18 22 26 30 34 38 42 46 51
```
Output:
- Discovers the **Operational Knee Point** (first QP where detection retention drops below $90\%$).
- Generates a publication-grade 3-panel figure at `outputs/qp_analysis/qp_retention_curve.png`:
  1. *Detection Retention (%) vs. QP*
  2. *Reconstruction Quality (PSNR dB) vs. QP*
  3. *Stream Bandwidth (kbps) vs. QP*

### 4. Run Instant Demo
Don't have footage yet? Generate synthetic surveillance video with simulated shadows, headlight glare, and crossing pedestrians/vehicles automatically:
```bash
python -m cctv_pipeline demo
```

---

## Running Automated Tests

Run the full unit test suite covering preprocessing, LBSP descriptors, SuBSENSE background models, tripwire vector geometry, and knee-point calculation:
```bash
python -m unittest discover tests -v
```

---

## CDNet2014 Benchmark Metrics

The platform provides a standardized evaluation engine complying with the ChangeDetection.NET benchmark:
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F-Measure ($F_1$)**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Percentage of Wrong Classifications (PWC)**: $100 \times \frac{FN + FP}{TP + FN + FP + TN}$
- **False Positive Rate (FPR)**: $\frac{FP}{FP + TN}$

---

## License & Attribution

Developed for the research project *"Robust CCTV Video Analytics Under Illumination Changes and Compression Artifacts"*, Department of Computer Science and Data Science, CHRIST (Deemed to be University).
Licensed under the MIT License.
