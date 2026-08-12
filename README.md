<div align="center">
  
# 🫀 Personalised Atrial Fibrillation (AF) Prediction System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![IoT](https://img.shields.io/badge/IoT-ThingSpeak-orange.svg)](#)
[![Hardware](https://img.shields.io/badge/Hardware-Arduino_Uno-teal.svg)](#)

*A real-time, edge-deployed atrial fibrillation prediction system achieving state-of-the-art accuracy through personalised Heart Rate Variability (HRV) modelling.*

**Developed by: Harikumaran Ramesh**

</div>

---

## 📖 Project Overview

This project implements a highly accurate, real-time atrial fibrillation prediction system. By leveraging personalised HRV modelling on the LTAF database, the system achieves an impressive **AUC of 0.930**. Furthermore, this solution goes beyond purely theoretical ML models by including a complete edge deployment pipeline utilizing an **Arduino Uno**, an **AD8232 ECG sensor**, and a live **ThingSpeak IoT cloud dashboard**.

## ✨ Key Results

Our personalised calibration approach significantly outperforms traditional population-based models:

| Metric | Result | Note |
|:---|:---:|:---|
| **Population Model (LOPO-CV)** | 0.657 AUC | Baseline performance |
| **Personalised Model (6h calib)** | **0.930 AUC** | **+0.273 improvement** |
| **High Accuracy Patients** | 19/21 | Achieved AUC ≥ 0.80 |
| **Exceptional Accuracy Patients** | 16/21 | Achieved AUC ≥ 0.90 |
| **Real-Time Alert (Record 06)** | -26 mins | Alert triggered 26 min before AF onset |
| **Ablation Study** | 46 variants | Rigorously tested architectures |

## 🚀 Novel Contributions

1. **Cardiac Phenotype Discovery:** Identified two distinct pre-AF cardiac phenotypes using Kendall τ.
2. **Leakage Quantification:** Rigorously quantified data leakage in existing published literature (~0.25 AUC inflation).
3. **Novel Rhythm Features:** Introduced 3 novel features: *Regularity Index*, *RR Monotonicity*, and *Variance Stability Ratio*.
4. **Complete Edge Deployment:** Full translation from Python ML pipelines to C++ Arduino sketches with ThingSpeak IoT integration.

## 📂 Repository Structure

```text
├── 01_Pipeline_Code/     # Full Python pipeline (steps 1–6 for ML model)
├── 02_Edge_Deployment/   # Arduino C++ sketches & Raspberry Pi integration
├── 03_Documents/         # Research paper, ablation study & workflow documentation
└── 04_Visualisation/     # Figure and graph generation scripts
```

## ⚙️ How to Run (PC Environment)

1. **Install Dependencies:**
   ```bash
   pip install wfdb numpy scipy scikit-learn joblib antropy imbalanced-learn
   ```
2. **Download Dataset:** Obtain the **LTAF database** from PhysioNet.
3. **Execute Pipeline:** Run steps 1–5 in sequential order from the `01_Pipeline_Code/` directory.
4. **Visualise Results:** Execute `visualise_results.py` in the `04_Visualisation/` folder to generate all figures.

## 📡 Edge Deployment (Arduino + ThingSpeak)

To deploy the model on hardware for live IoT monitoring:
- Please refer to the detailed guide at: [`02_Edge_Deployment/thingspeak_setup_guide.md`](02_Edge_Deployment/thingspeak_setup_guide.md)

## 📊 Databases Used

- **Long-Term AF Database (PhysioNet)** — Main evaluation dataset
- **MIT-BIH Arrhythmia (PhysioNet)** — Negative control dataset

---
*Created and maintained by [Harikumaran Ramesh](https://github.com/)*
