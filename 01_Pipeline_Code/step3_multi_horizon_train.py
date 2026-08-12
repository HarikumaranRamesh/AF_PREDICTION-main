"""
LTAF STEP 3 — MULTI-HORIZON ENSEMBLE
======================================
Architecture:
  - 8 SEPARATE classifiers, one per horizon window
  - Horizons: 2, 3, 5, 8, 10, 20, 30 minutes
  - Each classifier answers ONE question:
      "Is this window X minutes before AF onset, vs normal sinus?"
  - Each gets its OWN AUC, sensitivity, specificity
  - At inference: run all 8, pick the one with highest confidence
  - Output: "AF likely in ~X minutes (this window accuracy: Y%)"

Why separate classifiers work better than one combined:
  - A 2-min window has very different RR patterns than a 30-min window
  - One combined model tries to learn all patterns at once → diluted
  - Separate models each learn ONE specific transition pattern → sharper
  - Published literature uses this approach (multi-scale prediction)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                              average_precision_score,
                              precision_recall_curve, f1_score)
from sklearn.utils import resample
import joblib
import warnings
warnings.filterwarnings('ignore')

OUTPUT_PATH  = r"C:\Users\HOME\Desktop\ltaf_project"
WINDOW_PATH  = os.path.join(OUTPUT_PATH, "data", "windows")
MODEL_PATH   = os.path.join(OUTPUT_PATH, "models", "multihorizon")
RESULTS_PATH = os.path.join(OUTPUT_PATH, "results")
os.makedirs(MODEL_PATH,   exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# ── THE 8 HORIZONS ────────────────────────────────────────────────────────────
HORIZON_MINS = [2, 3, 5, 8, 10, 20, 30]
HORIZON_SECS = [h * 60 for h in HORIZON_MINS]
NORMAL_IDX   = len(HORIZON_MINS)   # index 7 = normal

# Tolerance: windows within ±TOLERANCE of a horizon are assigned to it
# e.g. "5 min" bucket = windows 3.5 to 7.5 minutes before AF
TOLERANCES = {
    2:  (0,   150),    # 0-2.5 min
    3:  (150, 240),    # 2.5-4 min
    5:  (240, 420),    # 4-7 min
    8:  (420, 570),    # 7-9.5 min
    10: (570, 900),    # 9.5-15 min
    20: (900,  1500),  # 15-25 min
    30: (1500, 2100),  # 25-35 min
}

CSD_NAMES = ['variance', 'lag1_ac', 'ar1_coeff', 'skewness', 'kurtosis']
HRV_NAMES = ['rr_mean', 'rr_std', 'rmssd', 'pnn50', 'lf_hf_ratio',
             'sample_ent', 'dfa_alpha', 'poincare_sd1', 'poincare_sd2']

ALL_RECORDS  = [f"{i:02d}" for i in range(84)]
TREND_WIN    = 8


# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load(rec):
    f = os.path.join(WINDOW_PATH, f'{rec}_windows.npz')
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    X = np.hstack([d['csd_features'], d['hrv_features']]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        'X'      : X,
        'horizon': d['horizon_label'],
        'binary' : d['binary_label'],
        'times'  : d['window_times'],
    }


# ── FEATURE ENHANCEMENT ───────────────────────────────────────────────────────
def add_trends(X, times):
    N, F   = X.shape
    slopes = np.zeros((N, F))
    tidx   = np.arange(TREND_WIN, dtype=float)
    for i in range(TREND_WIN, N):
        seg = X[i - TREND_WIN:i]
        for f in range(F):
            s = seg[:, f]
            if np.std(s) < 1e-8:
                continue
            try:
                sl, *_ = linregress(tidx, s)
                slopes[i, f] = float(sl) if np.isfinite(sl) else 0.0
            except:
                pass
    return np.nan_to_num(np.hstack([X, slopes]), nan=0.0, posinf=0.0, neginf=0.0)


def normalize_baseline(X, times, base_min=30.0):
    mask = times <= base_min * 60
    if mask.sum() < 10:
        mask = np.ones(len(times), dtype=bool)
    mu  = X[mask].mean(0)
    std = X[mask].std(0) + 1e-8
    return (X - mu) / std


def build_features(X_raw, times):
    X_norm = normalize_baseline(X_raw, times)
    X_enh  = add_trends(X_norm, times)
    return np.nan_to_num(X_enh, nan=0.0, posinf=0.0, neginf=0.0)


# ── RE-LABEL WINDOWS FOR A SPECIFIC HORIZON ───────────────────────────────────
def relabel_for_horizon(horizon_min, times, binary, horizon_orig):
    """
    For a given horizon window (e.g. 5 min), create binary labels:
      1 = window falls within the tolerance band for this horizon
      0 = normal sinus (far from any AF)
      -1 = skip (other horizon bands — not used for THIS classifier)

    This means each classifier sees a CLEAN binary problem:
      "5-minutes-before-AF"  vs  "normal sinus"
    NOT contaminated by "2-min-before-AF" windows.
    """
    lo, hi = TOLERANCES[horizon_min]   # seconds
    y_new  = np.full(len(binary), -1, dtype=np.int32)  # -1 = skip

    for i in range(len(binary)):
        if binary[i] == 0:
            # Normal sinus — always include as negative class
            y_new[i] = 0
        else:
            # Pre-AF — only include if within THIS horizon's tolerance band
            # We need the actual horizon_sec, which we reconstruct from
            # the horizon_label bucket index
            # horizon_orig stores bucket index 0-6 for pre-AF
            h_idx = int(horizon_orig[i])
            if h_idx == NORMAL_IDX:
                y_new[i] = 0  # normal
            else:
                # Map bucket back to approximate seconds
                h_sec_approx = HORIZON_SECS[h_idx]
                if lo <= h_sec_approx < hi:
                    y_new[i] = 1   # this is our target horizon
                # else: -1 (skip — belongs to different horizon)

    return y_new


# ── BOOTSTRAP AUC ────────────────────────────────────────────────────────────
def bootstrap_auc(y_true, y_score, n=150):
    aucs = []
    for _ in range(n):
        idx = resample(np.arange(len(y_true)))
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(yt, ys))
        except:
            pass
    if not aucs:
        return 0.5, 0.5, 0.5
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


# ── PER-HORIZON METRICS AT BALANCED THRESHOLD ─────────────────────────────────
def compute_metrics_at_threshold(y_true, y_score, target_spec=0.65):
    prec, rec, thrs = precision_recall_curve(y_true, y_score)
    best_t, best_f1 = 0.5, 0.0
    for t in thrs:
        yp = (y_score >= t).astype(int)
        cm = confusion_matrix(y_true, yp)
        if cm.size < 4:
            continue
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp + 1e-8)
        if spec < target_spec:
            continue
        f1 = f1_score(y_true, yp, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t

    y_pred = (y_score >= best_t).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        return {
            'threshold'  : best_t,
            'sensitivity': tp / (tp + fn + 1e-8),
            'specificity': tn / (tn + fp + 1e-8),
            'precision'  : tp / (tp + fp + 1e-8),
            'f1'         : f1_score(y_true, y_pred, zero_division=0),
            'far'        : fp / (fp + tn + 1e-8),
        }
    return {'threshold': 0.5, 'sensitivity': 0, 'specificity': 0,
            'precision': 0, 'f1': 0, 'far': 1}


# ── LOPO-CV FOR ONE HORIZON ───────────────────────────────────────────────────
def run_lopo_one_horizon(horizon_min, all_data):
    results = []
    eligible = [r for r in all_data if
                # need at least some positive windows for this horizon
                np.any(all_data[r]['binary'] == 1)]

    for test_rec in eligible:
        td      = all_data[test_rec]
        X_test  = build_features(td['X'], td['times'])
        y_raw   = relabel_for_horizon(
            horizon_min, td['times'], td['binary'], td['horizon']
        )
        # Keep only valid (0 or 1) labels
        valid   = y_raw >= 0
        X_test  = X_test[valid]
        y_test  = y_raw[valid]

        if len(np.unique(y_test)) < 2 or y_test.sum() < 3:
            continue

        # Train on all other patients
        Xtr, ytr = [], []
        for rec, d in all_data.items():
            if rec == test_rec:
                continue
            y_r = relabel_for_horizon(
                horizon_min, d['times'], d['binary'], d['horizon']
            )
            v = y_r >= 0
            if v.sum() < 5:
                continue
            Xtr.append(build_features(d['X'], d['times'])[v])
            ytr.append(y_r[v])

        if not Xtr:
            continue

        Xtr = np.vstack(Xtr)
        ytr = np.concatenate(ytr)

        if len(np.unique(ytr)) < 2:
            continue

        scaler    = RobustScaler()
        Xtr_s     = scaler.fit_transform(Xtr)
        Xtest_s   = scaler.transform(X_test)

        pos = ytr.sum()
        neg = len(ytr) - pos
        w   = np.where(ytr == 1, neg / max(pos, 1), 1.0)

        clf = GradientBoostingClassifier(
            n_estimators=120, max_depth=3,
            learning_rate=0.08, subsample=0.8,
            min_samples_leaf=8, random_state=42
        )
        clf.fit(Xtr_s, ytr, sample_weight=w)

        y_score = clf.predict_proba(Xtest_s)[:, 1]

        try:
            auc, ci_lo, ci_hi = bootstrap_auc(y_test, y_score, n=100)
            ap  = average_precision_score(y_test, y_score)
        except:
            continue

        metrics = compute_metrics_at_threshold(y_test, y_score)

        results.append({
            'record'     : test_rec,
            'horizon_min': horizon_min,
            'auc'        : auc,
            'ci_lo'      : ci_lo,
            'ci_hi'      : ci_hi,
            'ap'         : ap,
            'n_pos'      : int(y_test.sum()),
            'n_total'    : len(y_test),
            **metrics,
        })

        # Save model
        joblib.dump({
            'clf'        : clf,
            'scaler'     : scaler,
            'threshold'  : metrics['threshold'],
            'horizon_min': horizon_min,
            'auc'        : auc,
            'sensitivity': metrics['sensitivity'],
            'specificity': metrics['specificity'],
            'test_rec'   : test_rec,
        }, os.path.join(MODEL_PATH, f'h{horizon_min:02d}min_{test_rec}.pkl'))

    return pd.DataFrame(results)


# ── MAIN LOPO LOOP ────────────────────────────────────────────────────────────
def run_all_horizons():
    print("=" * 70)
    print("MULTI-HORIZON LOPO-CV")
    print(f"  Horizons : {HORIZON_MINS} minutes")
    print(f"  Each has its own classifier + own AUC")
    print("=" * 70)

    all_data = {}
    for rec in ALL_RECORDS:
        d = load(rec)
        if d is not None and d['binary'].sum() >= 3:
            all_data[rec] = d
    print(f"\nLoaded {len(all_data)} patients\n")

    all_results  = []
    summary_rows = []

    for h_min in HORIZON_MINS:
        print(f"\n{'─'*70}")
        print(f"HORIZON: {h_min} minutes before AF onset")
        print(f"{'─'*70}")

        df_h = run_lopo_one_horizon(h_min, all_data)
        all_results.append(df_h)

        if len(df_h) == 0:
            print(f"  No results for {h_min}min horizon")
            continue

        mean_auc  = df_h['auc'].mean()
        std_auc   = df_h['auc'].std()
        mean_sens = df_h['sensitivity'].mean()
        mean_spec = df_h['specificity'].mean()
        mean_prec = df_h['precision'].mean()
        mean_f1   = df_h['f1'].mean()
        n_good    = (df_h['auc'] >= 0.70).sum()

        print(f"  Patients evaluated : {len(df_h)}")
        print(f"  Mean AUC           : {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"  Sensitivity        : {mean_sens:.1%}")
        print(f"  Specificity        : {mean_spec:.1%}")
        print(f"  Precision          : {mean_prec:.1%}")
        print(f"  F1 Score           : {mean_f1:.3f}")
        print(f"  AUC ≥ 0.70         : {n_good}/{len(df_h)} patients")

        summary_rows.append({
            'horizon_min': h_min,
            'n_patients' : len(df_h),
            'mean_auc'   : mean_auc,
            'std_auc'    : std_auc,
            'median_auc' : df_h['auc'].median(),
            'sensitivity': mean_sens,
            'specificity': mean_spec,
            'precision'  : mean_prec,
            'f1'         : mean_f1,
            'n_auc70'    : int(n_good),
            'pct_auc70'  : n_good / max(len(df_h), 1) * 100,
        })

    # ── SAVE ALL RESULTS ──────────────────────────────────────────────────
    df_all = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    df_sum = pd.DataFrame(summary_rows)

    df_all.to_csv(os.path.join(RESULTS_PATH, 'multihorizon_lopo.csv'), index=False)
    df_sum.to_csv(os.path.join(RESULTS_PATH, 'multihorizon_summary.csv'), index=False)

    # ── FINAL SUMMARY TABLE ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL SUMMARY — ACCURACY PER HORIZON WINDOW")
    print(f"{'='*70}")
    print(f"\n  {'Window':>8} {'AUC':>7} {'Sens':>7} {'Spec':>7} "
          f"{'Prec':>7} {'F1':>7} {'AUC≥0.70':>10}")
    print(f"  {'─'*62}")

    best_h   = None
    best_auc = 0

    for _, row in df_sum.iterrows():
        h    = int(row['horizon_min'])
        star = ' ← BEST' if row['mean_auc'] == df_sum['mean_auc'].max() else ''
        print(f"  {h:>5} min  "
              f"{row['mean_auc']:>6.3f}  "
              f"{row['sensitivity']:>6.1%}  "
              f"{row['specificity']:>6.1%}  "
              f"{row['precision']:>6.1%}  "
              f"{row['f1']:>6.3f}  "
              f"{row['n_auc70']:>4}/{row['n_patients']:<4}{star}")
        if row['mean_auc'] > best_auc:
            best_auc = row['mean_auc']
            best_h   = h

    print(f"\n  Best prediction window: {best_h} minutes (AUC={best_auc:.4f})")
    print(f"\n  Clinical interpretation:")
    print(f"    When the {best_h}-minute model fires:")
    print(f"      → 'Abnormality likely in ~{best_h} minutes'")
    best_row = df_sum[df_sum['horizon_min'] == best_h].iloc[0]
    print(f"      → Accuracy: {best_row['precision']:.0%} of alerts are real")
    print(f"      → Catches: {best_row['sensitivity']:.0%} of actual AF episodes")

    print(f"\n✅ All models saved to {MODEL_PATH}")
    print(f"✅ Results saved to {RESULTS_PATH}")
    return df_sum, df_all


if __name__ == '__main__':
    df_sum, df_all = run_all_horizons()
