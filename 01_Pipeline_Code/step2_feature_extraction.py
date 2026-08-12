"""
LTAF STEP 2 — FIXED (Strict episode filtering)
================================================
Problem: Record 15 had 801 episodes, record 10 had 80 episodes.
These are sustained AF patients cycling in/out of AF every few minutes.
The model was trained on noise.

Fix: Only use AF episodes preceded by >= 30 min of clean sinus rhythm.
This leaves only genuine paroxysmal AF onsets where CSD can build.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, pearsonr
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')

OUTPUT_PATH  = r"C:\Users\HOME\Desktop\ltaf_project"
RR_PATH      = os.path.join(OUTPUT_PATH, "data", "rr")
WINDOW_PATH  = os.path.join(OUTPUT_PATH, "data", "windows")
os.makedirs(WINDOW_PATH, exist_ok=True)

WINDOW_BEATS         = 120
STEP_BEATS           = 10
MIN_BEATS            = 80
MIN_SINUS_BEFORE_SEC = 1800   # 30 min clean sinus required before AF onset
MIN_AF_DUR_SEC       = 60     # AF must last >= 60 seconds
MIN_USABLE_EPISODES  = 2      # need at least 2 good episodes per patient

HORIZON_MINS = [5, 10, 15, 20, 30, 45, 60]
HORIZON_SECS = [h * 60 for h in HORIZON_MINS]
HORIZON_BINS = [0] + HORIZON_SECS
NORMAL_IDX   = len(HORIZON_MINS)

CSD_NAMES = ['variance', 'lag1_ac', 'ar1_coeff', 'skewness', 'kurtosis']
HRV_NAMES = ['rr_mean', 'rr_std', 'rmssd', 'pnn50', 'lf_hf_ratio',
             'sample_ent', 'dfa_alpha', 'poincare_sd1', 'poincare_sd2']

ALL_RECORDS = [f"{i:02d}" for i in range(84)]


def safe(func, arr, fallback=0.0):
    try:
        v = float(func(arr))
        return v if np.isfinite(v) else fallback
    except:
        return fallback

def safe_lag1(arr):
    try:
        if len(arr) < 4 or np.std(arr) < 1e-8: return 0.0
        r, _ = pearsonr(arr[:-1], arr[1:])
        return float(r) if np.isfinite(r) else 0.0
    except:
        return 0.0

def safe_ar1(arr):
    try:
        if len(arr) < 4 or np.std(arr) < 1e-8: return 0.0
        y = arr[1:]
        X = np.column_stack([arr[:-1], np.ones(len(arr)-1)])
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(c[0]) if np.isfinite(c[0]) else 0.0
    except:
        return 0.0

def safe_entropy(arr):
    try:
        from antropy import sample_entropy
        if len(arr) < 12 or np.std(arr) < 1e-8: return 0.0
        v = sample_entropy(arr, order=2)
        return float(v) if np.isfinite(v) else 0.0
    except:
        return 0.0

def safe_dfa(arr):
    try:
        from antropy import detrended_fluctuation
        if len(arr) < 20 or np.std(arr) < 1e-8: return 1.0
        v = detrended_fluctuation(arr)
        return float(v) if np.isfinite(v) else 1.0
    except:
        return 1.0

def safe_lf_hf(rr):
    try:
        if len(rr) < 16: return 1.0
        rr_d = rr - np.mean(rr)
        f, p = welch(rr_d, fs=1.0, nperseg=min(len(rr_d), 64))
        lf = np.trapz(p[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)])
        hf = np.trapz(p[(f >= 0.15) & (f <= 0.40)], f[(f >= 0.15) & (f <= 0.40)])
        r  = lf / (hf + 1e-9)
        return float(r) if np.isfinite(r) else 1.0
    except:
        return 1.0

def compute_features(rr):
    rr   = rr.astype(np.float64)
    diff = np.diff(rr)
    csd  = {
        'variance'  : safe(np.var, rr),
        'lag1_ac'   : safe_lag1(rr),
        'ar1_coeff' : safe_ar1(rr),
        'skewness'  : safe(skew, rr),
        'kurtosis'  : safe(kurtosis, rr),
    }
    hrv  = {
        'rr_mean'     : safe(np.mean, rr),
        'rr_std'      : safe(np.std,  rr),
        'rmssd'       : safe(lambda x: np.sqrt(np.mean(x**2)), diff),
        'pnn50'       : safe(lambda x: np.mean(np.abs(x) > 50), diff),
        'lf_hf_ratio' : safe_lf_hf(rr),
        'sample_ent'  : safe_entropy(rr),
        'dfa_alpha'   : safe_dfa(rr),
        'poincare_sd1': safe(lambda x: np.std(x / np.sqrt(2)), diff),
        'poincare_sd2': float(np.std(rr[:-1]) * np.sqrt(2)) if len(rr) > 2 else 0.0,
    }
    return csd, hrv


def get_usable_episodes(all_af_episodes):
    usable = []
    for i, (af_start, af_end) in enumerate(all_af_episodes):
        if (af_end - af_start) < MIN_AF_DUR_SEC:
            continue
        sinus_before = af_start if i == 0 else af_start - all_af_episodes[i-1][1]
        if sinus_before >= MIN_SINUS_BEFORE_SEC:
            usable.append((af_start, af_end))
    return usable


def label_window(t_end, usable_episodes, post_af_buffer=120, pre_af_skip=30):
    if not usable_episodes:
        return 0, np.inf, NORMAL_IDX

    for af_start, af_end in usable_episodes:
        if af_start <= t_end <= af_end:
            return -1, -1, -1
        if af_end < t_end < af_end + post_af_buffer:
            return -1, -1, -1

    future = [(s, e) for s, e in usable_episodes if s > t_end]
    if not future:
        past_ends = [e for _, e in usable_episodes if e <= t_end]
        if past_ends and (t_end - max(past_ends)) < post_af_buffer:
            return -1, -1, -1
        return 0, np.inf, NORMAL_IDX

    h_sec = future[0][0] - t_end
    if h_sec < pre_af_skip:
        return -1, -1, -1

    if h_sec <= HORIZON_SECS[-1]:
        for i in range(len(HORIZON_BINS) - 1):
            if HORIZON_BINS[i] <= h_sec < HORIZON_BINS[i+1]:
                return 1, h_sec, i
        return 1, h_sec, len(HORIZON_MINS) - 1

    past_ends = [e for _, e in usable_episodes if e <= t_end]
    if past_ends and (t_end - max(past_ends)) < post_af_buffer:
        return -1, -1, -1
    return 0, h_sec, NORMAL_IDX


def process_record(rec):
    rr_file = os.path.join(RR_PATH, f'{rec}_rr.npz')
    if not os.path.exists(rr_file):
        return False, "no rr file"

    d        = np.load(rr_file, allow_pickle=True)
    rr_ms    = d['rr_ms']
    rr_times = d['rr_times']
    par_raw  = d['paroxysmal_af']

    if par_raw.shape[0] == 0:
        return False, "no paroxysmal AF"

    all_episodes = [(row[0], row[1]) for row in par_raw]
    usable       = get_usable_episodes(all_episodes)

    if len(usable) < MIN_USABLE_EPISODES:
        return False, (f"only {len(usable)} usable episodes "
                       f"(need {MIN_USABLE_EPISODES}) — sustained/rapid-cycling AF")

    if len(rr_ms) < WINDOW_BEATS * 2:
        return False, "too short"

    csd_wins, hrv_wins = [], []
    horizon_lbls, win_times, binary_lbls = [], [], []
    n_wins = n_skip = 0

    for start in range(0, len(rr_ms) - WINDOW_BEATS, STEP_BEATS):
        end       = start + WINDOW_BEATS
        rr_window = rr_ms[start:end]
        t_end     = rr_times[end - 1]
        t_center  = (rr_times[start] + t_end) / 2.0

        if (len(rr_window) < MIN_BEATS or
                np.any(~np.isfinite(rr_window)) or
                np.std(rr_window) < 1e-6):
            n_skip += 1
            continue

        label, h_sec, bucket = label_window(t_end, usable)
        if label == -1:
            n_skip += 1
            continue

        csd_f, hrv_f = compute_features(rr_window)
        if not all(np.isfinite(v) for v in list(csd_f.values()) + list(hrv_f.values())):
            n_skip += 1
            continue

        csd_wins.append([csd_f[k] for k in CSD_NAMES])
        hrv_wins.append([hrv_f[k] for k in HRV_NAMES])
        horizon_lbls.append(bucket)
        win_times.append(t_center)
        binary_lbls.append(label)
        n_wins += 1

    if n_wins == 0:
        return False, "no valid windows"

    np.savez_compressed(
        os.path.join(WINDOW_PATH, f'{rec}_windows.npz'),
        csd_features  = np.array(csd_wins,     dtype=np.float32),
        hrv_features  = np.array(hrv_wins,      dtype=np.float32),
        horizon_label = np.array(horizon_lbls,  dtype=np.int32),
        window_times  = np.array(win_times,     dtype=np.float32),
        binary_label  = np.array(binary_lbls,   dtype=np.int32),
        csd_names     = np.array(CSD_NAMES),
        hrv_names     = np.array(HRV_NAMES),
    )

    n1 = int(np.sum(binary_lbls))
    return True, {
        'n_windows'      : n_wins,
        'n_pre_af'       : n1,
        'n_normal'       : n_wins - n1,
        'pct_pre'        : n1 / max(n_wins, 1) * 100,
        'n_all_episodes' : len(all_episodes),
        'n_usable'       : len(usable),
    }


if __name__ == '__main__':
    print("=" * 65)
    print("LTAF STEP 2 (FIXED) — Strict Episode Filtering")
    print(f"  Min sinus before AF : {MIN_SINUS_BEFORE_SEC//60} min")
    print(f"  Min AF duration     : {MIN_AF_DUR_SEC} sec")
    print(f"  Min usable episodes : {MIN_USABLE_EPISODES}")
    print("=" * 65)

    totals     = {'wins': 0, 'pre': 0, 'norm': 0, 'patients': 0}
    skip_counts = {}

    for i, rec in enumerate(ALL_RECORDS):
        print(f"[{i+1:02d}/84] {rec}...", end=' ', flush=True)
        ok, res = process_record(rec)
        if ok:
            totals['wins']     += res['n_windows']
            totals['pre']      += res['n_pre_af']
            totals['norm']     += res['n_normal']
            totals['patients'] += 1
            flag = '✅' if 5 <= res['pct_pre'] <= 30 else '⚠️ '
            print(f"{flag} {res['n_windows']} wins | "
                  f"{res['n_usable']}/{res['n_all_episodes']} usable | "
                  f"{res['pct_pre']:.1f}% pre-AF")
        else:
            print(f"— {res}")
            key = res[:40]
            skip_counts[key] = skip_counts.get(key, 0) + 1

    pct = totals['pre'] / max(totals['wins'], 1) * 100
    print(f"\n{'='*65}")
    print(f"FINAL BALANCE")
    print(f"  Patients kept : {totals['patients']}")
    print(f"  Total windows : {totals['wins']:,}")
    print(f"  Pre-AF        : {totals['pre']:,} ({pct:.1f}%)")
    print(f"  Normal sinus  : {totals['norm']:,} ({100-pct:.1f}%)")
    print(f"  Ratio N:P     : {totals['norm']/max(totals['pre'],1):.1f}:1")
    if 5 <= pct <= 25:
        print(f"\n✅  Good balance. Run ltaf_step3_train.py next.")
        print(f"    Expected AUC: 0.65-0.80 (vs 0.54 before fix)")
    else:
        print(f"\n⚠️   Adjust MIN_SINUS_BEFORE_SEC if balance is off")
