============================================================
 PERSONALISED AF PREDICTION SYSTEM
 Submitted by: Harikumaran Ramesh
============================================================

PROJECT SUMMARY:
  Real-time atrial fibrillation prediction system achieving
  AUC 0.930 using personalised HRV modelling on the LTAF
  database. Deployed on Arduino Uno + AD8232 ECG sensor
  with live ThingSpeak IoT cloud dashboard.

KEY RESULTS:
  Population model (LOPO-CV):   AUC = 0.657
  Personalised model (6h calib): AUC = 0.930  (+0.273)
  19/21 patients:                AUC ≥ 0.80
  16/21 patients:                AUC ≥ 0.90
  Record 06 real-time alert:     26 min before AF onset
  Ablation study:                46 variants tested

NOVEL CONTRIBUTIONS:
  1. Two pre-AF cardiac phenotypes discovered (Kendall τ)
  2. Data leakage quantified in published literature (~0.25 AUC)
  3. 3 novel rhythm features: Regularity Index, RR Monotonicity,
     Variance Stability Ratio
  4. Full edge deployment: Arduino + ThingSpeak IoT

FOLDER STRUCTURE:
  01_Pipeline_Code/       — Full Python pipeline (steps 1–6)
  02_Edge_Deployment/     — Arduino sketches + Raspberry Pi code
  03_Documents/           — Paper, ablation study, workflow docs
  04_Visualisation/       — Figure generation script

HOW TO RUN (PC):
  1. Install: pip install wfdb numpy scipy scikit-learn joblib antropy imbalanced-learn
  2. Download LTAF database from PhysioNet
  3. Run steps 1–5 in order from 01_Pipeline_Code/
  4. Run visualise_results.py for all figures

HOW TO RUN (Arduino + ThingSpeak):
  See 02_Edge_Deployment/thingspeak_setup_guide.md

DATABASES USED:
  MIT-BIH Arrhythmia (PhysioNet) — negative control
  Long-Term AF Database (PhysioNet) — main dataset
============================================================
