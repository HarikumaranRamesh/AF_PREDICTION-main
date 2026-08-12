"""
LTAF STEP 1 — FIXED (Memory Error patch)
==========================================
Problem: Record 21 (and some others) are 24-25h recordings.
neurokit2 Pan-Tompkins tries to process the entire signal at once
-> MemoryError on the moving window average step.

Fix: Process ECG in 30-minute chunks, concatenate R-peaks.
Also adds a fallback method (hamilton) if pantompkins fails.
Also adds RESUME support: skips records already processed.
"""

import os
import numpy as np
import json
import wfdb
import neurokit2 as nk
import warnings
warnings.filterwarnings('ignore')

DATA_PATH   = r"C:\Users\HOME\Downloads\ltaf-database-1.0.0"
OUTPUT_PATH = r"C:\Users\HOME\Desktop\ltaf_project"
RR_PATH     = os.path.join(OUTPUT_PATH, "data", "rr")
os.makedirs(RR_PATH, exist_ok=True)

FS          = 128
CHUNK_SEC   = 1800      # 30 minutes per chunk — safe for any machine
CHUNK_SAMP  = CHUNK_SEC * FS
ALL_RECORDS = [f"{i:02d}" for i in range(84)]


def is_good_signal(ecg):
    if len(ecg) < FS * 60:           return False, "too short"
    if np.std(ecg) < 1e-6:           return False, "flat"
    if not np.all(np.isfinite(ecg)):  return False, "inf/nan"
    clipped = np.mean((ecg == ecg.max()) | (ecg == ecg.min()))
    if clipped > 0.05:                return False, f"clipped {clipped:.1%}"
    return True, "ok"


def extract_af_onsets(annotation, fs=FS):
    af_onsets, sinus_periods = [], []
    af_start = sinus_start = None
    current_rhythm = None

    for sample, symbol, aux in zip(annotation.sample,
                                    annotation.symbol,
                                    annotation.aux_note):
        t   = sample / fs
        rhy = aux.strip() if aux else None
        if rhy:
            if current_rhythm == '(AFIB' and af_start is not None:
                af_onsets.append((af_start, t))
                af_start = None
            elif current_rhythm in ('(N', '(SBR', '(J') and sinus_start is not None:
                sinus_periods.append((sinus_start, t))
                sinus_start = None
            current_rhythm = rhy
            if rhy == '(AFIB':
                af_start = t
            elif rhy in ('(N', '(SBR', '(J'):
                sinus_start = t

    end = annotation.sample[-1] / fs if len(annotation.sample) > 0 else 0
    if current_rhythm == '(AFIB' and af_start is not None:
        af_onsets.append((af_start, end))
    elif current_rhythm in ('(N', '(SBR', '(J') and sinus_start is not None:
        sinus_periods.append((sinus_start, end))

    return af_onsets, sinus_periods


def extract_rr_chunked(ecg_raw, fs=FS):
    """
    Process ECG in 30-min chunks to avoid MemoryError.
    Falls back from pantompkins -> hamilton -> neurokit if needed.
    """
    n_total = len(ecg_raw)
    all_r_times = []

    n_chunks = (n_total // CHUNK_SAMP) + 1
    for ci, chunk_start in enumerate(range(0, n_total, CHUNK_SAMP)):
        chunk_end  = min(chunk_start + CHUNK_SAMP, n_total)
        chunk      = ecg_raw[chunk_start:chunk_end].copy()
        offset_sec = chunk_start / fs

        if len(chunk) < fs * 10:
            continue

        # Replace any inf/nan
        chunk = np.where(np.isfinite(chunk), chunk, 0.0)

        r_times_chunk = None
        for method in ['pantompkins1985', 'hamilton', 'neurokit']:
            try:
                ecg_c = nk.ecg_clean(chunk, sampling_rate=fs, method=method)
                _, rp  = nk.ecg_peaks(ecg_c, sampling_rate=fs, method=method)
                peaks  = rp['ECG_R_Peaks']
                if len(peaks) > 5:
                    r_times_chunk = peaks / fs + offset_sec
                    break
            except Exception:
                continue

        if r_times_chunk is not None and len(r_times_chunk) > 0:
            all_r_times.append(r_times_chunk)

    if not all_r_times:
        return None, None

    r_times = np.concatenate(all_r_times)
    r_times = np.sort(np.unique(r_times))   # deduplicate chunk boundaries

    if len(r_times) < 20:
        return None, None

    rr_ms    = np.diff(r_times) * 1000.0
    rr_times = r_times[1:]

    valid    = (rr_ms >= 300) & (rr_ms <= 2000)
    rr_ms    = rr_ms[valid]
    rr_times = rr_times[valid]

    if len(rr_ms) < 20:
        return None, None

    # Ectopic correction
    rr_corr = rr_ms.copy()
    w = 7
    for i in range(w, len(rr_corr) - w):
        med = np.median(rr_corr[i-w:i+w])
        if abs(rr_corr[i] - med) / (med + 1e-8) > 0.25:
            rr_corr[i] = (rr_corr[i-1] + rr_corr[i+1]) / 2.0

    return rr_corr, rr_times


def process_record(rec):
    path = os.path.join(DATA_PATH, rec)
    if not os.path.exists(path + '.dat'):
        return False, "missing"

    try:
        record = wfdb.rdrecord(path)
        ann    = wfdb.rdann(path, 'atr')
    except Exception as e:
        return False, str(e)

    ecg_raw = record.p_signal[:, 0].astype(np.float64)
    ecg_raw = np.where(np.isfinite(ecg_raw), ecg_raw, 0.0)

    ok, why = is_good_signal(ecg_raw)
    if not ok:
        return False, why

    n_chunks = len(ecg_raw) // CHUNK_SAMP + 1
    print(f"  [{len(ecg_raw)/FS/3600:.1f}h, {n_chunks} chunks]", end=' ', flush=True)

    rr_ms, rr_times = extract_rr_chunked(ecg_raw, FS)
    if rr_ms is None:
        return False, "R-peak detection failed"

    af_onsets, sinus_periods = extract_af_onsets(ann, FS)

    paroxysmal_af = [(s, e) for s, e in af_onsets
                     if e < record.sig_len / FS - 60]

    info = {
        'record'         : rec,
        'duration_hours' : record.sig_len / FS / 3600,
        'n_rr'           : len(rr_ms),
        'n_af_episodes'  : len(af_onsets),
        'n_paroxysmal'   : len(paroxysmal_af),
        'n_sinus_periods': len(sinus_periods),
        'total_af_hours' : sum(e - s for s, e in af_onsets) / 3600,
        'mean_hr'        : float(60000 / np.mean(rr_ms)),
        'fs'             : FS,
    }

    np.savez_compressed(
        os.path.join(RR_PATH, f'{rec}_rr.npz'),
        rr_ms         = rr_ms,
        rr_times      = rr_times,
        af_onsets     = np.array(af_onsets)     if af_onsets     else np.zeros((0, 2)),
        sinus_periods = np.array(sinus_periods) if sinus_periods else np.zeros((0, 2)),
        paroxysmal_af = np.array(paroxysmal_af) if paroxysmal_af else np.zeros((0, 2)),
    )
    with open(os.path.join(RR_PATH, f'{rec}_meta.json'), 'w') as f:
        json.dump(info, f, indent=2)

    return True, info


if __name__ == '__main__':
    import pandas as pd

    print("=" * 65)
    print("LTAF STEP 1 (FIXED) — Chunked RR Extraction")
    print(f"  Chunk size : {CHUNK_SEC // 60} min per chunk (memory safe)")
    print(f"  Fallback   : pantompkins -> hamilton -> neurokit")
    print(f"  Resume     : skips already-processed records")
    print("=" * 65)

    # Resume support — find already processed records
    done = set()
    for fname in os.listdir(RR_PATH):
        if fname.endswith('_rr.npz'):
            done.add(fname.replace('_rr.npz', ''))
    if done:
        print(f"  Resuming — {len(done)} records already done, skipping them\n")

    summary, failed = [], []

    for i, rec in enumerate(ALL_RECORDS):
        # Reload already-done records into summary
        if rec in done:
            meta_f = os.path.join(RR_PATH, f'{rec}_meta.json')
            if os.path.exists(meta_f):
                with open(meta_f) as f:
                    summary.append(json.load(f))
            print(f"[{i+1:02d}/84] Record {rec}... ⏭  (already done)")
            continue

        print(f"[{i+1:02d}/84] Record {rec}...", end=' ', flush=True)

        try:
            ok, result = process_record(rec)
        except MemoryError:
            ok, result = False, "MemoryError (even with chunks — skip)"
        except Exception as e:
            ok, result = False, f"Error: {e}"

        if ok:
            r = result
            print(f"✅ {r['duration_hours']:.1f}h | "
                  f"{r['n_af_episodes']} AF episodes | "
                  f"{r['n_paroxysmal']} paroxysmal | "
                  f"HR={r['mean_hr']:.0f}bpm")
            summary.append(r)
        else:
            if result != "missing":
                print(f"❌  {result}")
                failed.append(rec)
            else:
                print("(not found)")

    df = pd.DataFrame(summary)
    os.makedirs(os.path.join(OUTPUT_PATH, 'data'), exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_PATH, 'data', 'step1_summary.csv'), index=False)

    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"  Processed OK       : {len(summary)}")
    print(f"  Failed             : {len(failed)}")
    if failed:
        print(f"  Failed records     : {failed}")
    if len(df) > 0:
        par = df[df['n_paroxysmal'] > 0]
        print(f"  With paroxysmal AF : {len(par)}")
        print(f"  Mean duration      : {df['duration_hours'].mean():.1f}h")
        print(f"  Total AF episodes  : {df['n_af_episodes'].sum():.0f}")
        print(f"  Mean AF burden     : {df['total_af_hours'].mean():.2f}h/patient")
    print(f"\n✅ RR files saved to {RR_PATH}")
    print(f"   Run ltaf_step2_features.py next")
