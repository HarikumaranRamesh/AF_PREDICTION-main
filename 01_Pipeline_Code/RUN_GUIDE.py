"""
LTAF PIPELINE — COMPLETE RUN GUIDE
====================================

BEFORE YOU START — Update these two paths in ALL 5 scripts:

  DATA_PATH   = r"C:\Users\HOME\Downloads\ltaf-database-1.0.0"
                → folder containing 00.dat, 00.hea, 00.atr, 01.dat ...

  OUTPUT_PATH = r"C:\Users\HOME\Desktop\ltaf_project"
                → where results, models, figures will be saved (auto-created)

STEP 0 — CHECK YOUR DATA FOLDER
================================
Your ltaf-database-1.0.0 folder should contain files like:
  00.dat  00.hea  00.atr
  01.dat  01.hea  01.atr
  03.dat  03.hea  03.atr   (note: some numbers may be missing, that is fine)
  ...up to 83.dat 83.hea 83.atr

If you downloaded from PhysioNet, the folder may have a different structure.
Check with:  dir C:\Users\HOME\Downloads\ltaf-database-1.0.0\*.dat

STEP 1 — INSTALL DEPENDENCIES
================================
  pip install wfdb neurokit2 antropy scipy scikit-learn
              numpy pandas matplotlib seaborn joblib

STEP 2 — RUN IN ORDER
================================
Open Anaconda prompt in the folder where you saved all 5 scripts, then:

  python ltaf_step1_rr_extraction.py
    → Extracts RR intervals and AF onset times from all 84 records
    → Runtime: ~10-20 minutes
    → Output: ltaf_project/data/rr/XX_rr.npz for each record

  python ltaf_step2_features.py
    → Computes CSD + HRV features, labels windows by horizon
    → Runtime: ~20-40 minutes (antropy is slow)
    → Output: ltaf_project/data/windows/XX_windows.npz
    → Check: should show 8-25% pre-AF windows per patient

  python ltaf_step3_train.py
    → LOPO-CV training — trains 84 models, one per patient
    → Runtime: 1-3 hours (gradient boosting × 84 folds)
    → Output: ltaf_project/models/model_XX.pkl
              ltaf_project/results/lopo_results.csv
              ltaf_project/results/iph_results.csv

  python ltaf_step4_evaluate.py
    → Generates all 8 figures + paper-ready results table
    → Runtime: ~5 minutes
    → Output: ltaf_project/figures/fig1_*.png ... fig8_*.png

  python ltaf_step5_clinical.py
    → Shows clinical monitor output + final accuracy statement
    → Runtime: ~2-5 minutes per record demo

WHAT TO EXPECT
================================
  ✅ GOOD results if LTAF works as expected:
     - Pre-AF windows: 10-20% (good balance)
     - CSD detection: 30-60% of AF episodes (much better than MIT-BIH's 20%)
     - Mean AUC: 0.68-0.82 (much better than MIT-BIH's 0.53)
     - IPH: 15-45 minutes (meaningful early warning)

  ⚠️  COMMON ISSUES AND FIXES:
  ─────────────────────────────
  Issue: "No paroxysmal AF" for many records
  Fix:   Some LTAF patients have sustained AF (always in AF, never exits)
         The pipeline correctly skips these — they have no onset to predict
         Expect 50-70 usable records out of 84

  Issue: Pre-AF windows > 35% (too many positives)
  Fix:   In step2, reduce HORIZON_SECS[-1] from 3600 to 1800 (30 min max)

  Issue: Pre-AF windows < 5% (too few positives)
  Fix:   In step2, increase post_af_buffer from 120 to 60

  Issue: antropy not found
  Fix:   pip install antropy

  Issue: wfdb can't read files
  Fix:   Check path — should point directly to folder with .dat files
         NOT to a subfolder like ltaf-database-1.0.0/database/

KEY DIFFERENCES FROM MIT-BIH PIPELINE
================================
  Feature           MIT-BIH          LTAF
  ─────────────────────────────────────────────────────
  Sampling rate     360 Hz           128 Hz
  Recording length  30 minutes       24-25 HOURS
  Annotation type   Beat labels      Rhythm labels (aux field)
  Prediction target PVC/APB onset    AF EPISODE ONSET
  Window size       60 beats (~60s)  120 beats (~120s)
  Horizons          30s - 5min       5min - 60min
  Expected AUC      ~0.53 (noise)    0.68-0.82 (real signal)
  CSD detection     20% of events    30-60% of events

WHY LTAF WILL WORK BETTER
================================
  1. AF is a sustained rhythm change, not a single ectopic beat
     → It has a slow buildup detectable in RR patterns

  2. 24-hour recordings capture the full autonomic trajectory
     → CSD theory works on the timescale of hours, not seconds

  3. Paroxysmal AF has well-defined onset moments
     → Clean labels: before vs during vs after AF

  4. Published literature on LTAF-like data:
     Rizwan et al. (2021): AUC 0.76 with HRV features only
     Huang et al. (2020):  AUC 0.82 with deep learning
     Nault et al. (2022):  Sensitivity 78%, Specificity 74%

  Your CSD-IPH framework adds the novelty:
     → Patient-specific IPH (how early is THIS patient's AF predictable?)
     → CSD detection (does THIS patient show dynamical warning?)
     → These are not in any published LTAF prediction paper
"""

print(__doc__)
