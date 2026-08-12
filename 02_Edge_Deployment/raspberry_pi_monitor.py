"""
STEP 3 — COMPLETE EDGE AF MONITOR
====================================
Runs your trained personalised model in real time on Raspberry Pi.
Reads live ECG from AD8232 → detects R-peaks → computes RR intervals
→ extracts 42 features → runs your personal GBM classifier
→ fires graded AF alerts on screen (and optionally GPIO buzzer/LED).

HOW TO USE:
  1. Complete calibration first (6 hours sinus rhythm recording):
       python3 step3_monitor.py --mode calibrate --patient YOUR_NAME

  2. Run the monitor after calibration:
       python3 step3_monitor.py --mode predict --patient YOUR_NAME

  3. Or run both automatically (calibrate for 6h then predict):
       python3 step3_monitor.py --mode auto --patient YOUR_NAME

REQUIREMENTS:
  - step2_ecg_reader.py in same directory
  - joblib, sklearn, numpy, scipy, antropy installed (see step1_install.sh)
  - AD8232 wired correctly to Pi via MCP3008 (see wiring_and_setup.md)
"""

import os, sys, json, time, argparse, threading
import numpy as np
from collections import deque
from datetime import datetime
from scipy.stats import linregress, skew, kurtosis, pearsonr
from scipy.signal import welch
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Try hardware imports — graceful fallback for testing on non-Pi ───────────
try:
    import spidev
    import RPi.GPIO as GPIO
    from step2_ecg_reader import RPeakDetector, read_adc, check_leads
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    print("⚠  Hardware not available — running in SIMULATION MODE")
    print("   (uses saved LTAF RR data instead of live ECG)\n")

# ── PATHS ────────────────────────────────────────────────────────────────────
BASE_PATH    = os.path.expanduser("~/af_monitor")
MODEL_PATH   = os.path.join(BASE_PATH, "models")
DATA_PATH    = os.path.join(BASE_PATH, "recordings")
LOG_PATH     = os.path.join(BASE_PATH, "logs")
for d in [BASE_PATH, MODEL_PATH, DATA_PATH, LOG_PATH]:
    os.makedirs(d, exist_ok=True)

# ── HARDWARE PINS ─────────────────────────────────────────────────────────────
SAMPLE_RATE  = 500        # Hz
ADC_CHANNEL  = 0

# Optional alert hardware — set to None to disable
LED_RED_PIN  = 5          # GPIO pin for red LED  (CRITICAL)
LED_YLW_PIN  = 6          # GPIO pin for yellow LED (WARNING)
LED_GRN_PIN  = 13         # GPIO pin for green LED  (STABLE)
BUZZER_PIN   = 19         # GPIO pin for buzzer

# ── MONITOR SETTINGS ─────────────────────────────────────────────────────────
WINDOW_BEATS     = 120    # beats per feature window
STEP_BEATS       = 20     # slide step
TREND_WIN        = 8      # windows for trend slope
CALIBRATION_H    = 6.0    # hours of personal baseline
MIN_CALIB_BEATS  = 2000   # minimum beats before calibration is valid
EVAL_INTERVAL_S  = 30     # seconds between risk score evaluations
HORIZONS_MIN     = [5, 10, 20, 30]

ALERT_LEVELS = [
    (0.70, "🔴  CRITICAL",     "AF IMMINENT",        "RED"),
    (0.55, "⚠️   WARNING",      "AF PREDICTED SOON",  "YELLOW"),
    (0.40, "🟡  EARLY SIGNAL", "AF POSSIBLE",        "YELLOW"),
    (0.00, "✅  STABLE",        "NORMAL SINUS RHYTHM","GREEN"),
]

# ── FEATURE EXTRACTION (identical to your training pipeline) ─────────────────
def safe(f, a, fb=0.0):
    try: v = float(f(a)); return v if np.isfinite(v) else fb
    except: return fb

def lag1_ac(a):
    try:
        if len(a) < 4 or np.std(a) < 1e-8: return 0.0
        r, _ = pearsonr(a[:-1], a[1:]); return float(r) if np.isfinite(r) else 0.0
    except: return 0.0

def ar1_coef(a):
    try:
        if len(a) < 4 or np.std(a) < 1e-8: return 0.0
        y = a[1:]; X = np.column_stack([a[:-1], np.ones(len(a)-1)])
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(c[0]) if np.isfinite(c[0]) else 0.0
    except: return 0.0

def samp_ent(a):
    try:
        from antropy import sample_entropy
        if len(a) < 12 or np.std(a) < 1e-8: return 0.0
        v = sample_entropy(a, order=2); return float(v) if np.isfinite(v) else 0.0
    except: return 0.0

def dfa_alpha(a):
    try:
        from antropy import detrended_fluctuation
        if len(a) < 20 or np.std(a) < 1e-8: return 1.0
        v = detrended_fluctuation(a); return float(v) if np.isfinite(v) else 1.0
    except: return 1.0

def lf_hf_ratio(rr):
    try:
        if len(rr) < 16: return 1.0
        d = rr - np.mean(rr); f, p = welch(d, fs=1.0, nperseg=min(len(d), 64))
        lf = np.trapz(p[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)])
        hf = np.trapz(p[(f >= 0.15) & (f <= 0.40)], f[(f >= 0.15) & (f <= 0.40)])
        r  = float(lf / (hf + 1e-9))
        return r if np.isfinite(r) else 1.0
    except: return 1.0

def perm_ent(rr, order=3):
    try:
        if len(rr) < order + 1: return 0.0
        pats = {}
        for i in range(len(rr) - order + 1):
            p = tuple(np.argsort(rr[i:i+order])); pats[p] = pats.get(p, 0) + 1
        probs = np.array(list(pats.values()), dtype=float); probs /= probs.sum()
        from math import log, factorial
        me = log(factorial(order))
        e  = -sum(p * np.log(p + 1e-12) for p in probs)
        return float(e / me) if me > 0 else 0.0
    except: return 0.0

def regularity_idx(rr):
    if len(rr) < 4: return 0.0
    return float(np.mean(np.abs(np.diff(rr)) / (np.abs(rr[:-1]) + 1e-8) < 0.02))

def rr_mono(rr):
    if len(rr) < 4: return 0.0
    return float(np.sum(np.diff(rr) < 0) / (len(rr) - 1))

def var_stab(rr):
    if len(rr) < 40: return 1.0
    mid = len(rr) // 2
    return float(np.var(rr[mid:]) / (np.var(rr[:mid]) + 1e-8))

def compute_features(rr):
    """21 features — identical to training pipeline."""
    rr   = np.array(rr, dtype=np.float64)
    diff = np.diff(rr)
    v = np.array([
        safe(np.var, rr),                                          # 1
        lag1_ac(rr),                                               # 2
        ar1_coef(rr),                                              # 3
        safe(skew, rr),                                            # 4
        safe(kurtosis, rr),                                        # 5
        safe(np.mean, rr),                                         # 6
        safe(np.std, rr),                                          # 7
        safe(lambda x: np.sqrt(np.mean(x**2)), diff),             # 8 RMSSD
        safe(lambda x: np.mean(np.abs(x) > 50), diff),            # 9 pNN50
        lf_hf_ratio(rr),                                           # 10
        samp_ent(rr),                                              # 11
        dfa_alpha(rr),                                             # 12
        safe(lambda x: np.std(x / np.sqrt(2)), diff),             # 13 SD1
        float(np.std(rr[:-1]) * np.sqrt(2)) if len(rr)>2 else 0.0,# 14 SD2
        float(np.std(rr) / np.mean(rr)) if np.mean(rr)>1 else 0.0,# 15 CV
        perm_ent(rr),                                              # 16
        safe(lambda x: np.mean(np.abs(x)), diff),                  # 17
        float(np.percentile(rr, 95) - np.percentile(rr, 5)),      # 18
        regularity_idx(rr),                                        # 19 ★
        rr_mono(rr),                                               # 20 ★
        var_stab(rr),                                              # 21 ★
    ], dtype=np.float32)
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

def add_trends(X):
    """Add trend slopes for last TREND_WIN windows."""
    N, F  = X.shape
    slopes = np.zeros((N, F))
    tidx   = np.arange(TREND_WIN, dtype=float)
    for i in range(TREND_WIN, N):
        seg = X[i-TREND_WIN:i]
        for fi in range(F):
            s = seg[:, fi]
            if np.std(s) < 1e-8: continue
            try:
                sl, *_ = linregress(tidx, s)
                slopes[i, fi] = float(sl) if np.isfinite(sl) else 0.0
            except: pass
    return np.nan_to_num(np.hstack([X, slopes]), nan=0.0, posinf=0.0, neginf=0.0)


# ── GPIO ALERT OUTPUT ─────────────────────────────────────────────────────────
def setup_gpio_alerts():
    if not HARDWARE_AVAILABLE: return
    try:
        GPIO.setmode(GPIO.BCM)
        for pin in [LED_RED_PIN, LED_YLW_PIN, LED_GRN_PIN, BUZZER_PIN]:
            if pin: GPIO.setup(pin, GPIO.OUT); GPIO.output(pin, GPIO.LOW)
    except: pass

def fire_alert_gpio(level_color):
    """Flash the appropriate LED and buzzer based on alert level."""
    if not HARDWARE_AVAILABLE: return
    try:
        # Turn all off first
        for pin in [LED_RED_PIN, LED_YLW_PIN, LED_GRN_PIN]:
            if pin: GPIO.output(pin, GPIO.LOW)

        if level_color == "RED":
            if LED_RED_PIN: GPIO.output(LED_RED_PIN, GPIO.HIGH)
            if BUZZER_PIN:
                for _ in range(3):
                    GPIO.output(BUZZER_PIN, GPIO.HIGH); time.sleep(0.1)
                    GPIO.output(BUZZER_PIN, GPIO.LOW);  time.sleep(0.1)
        elif level_color == "YELLOW":
            if LED_YLW_PIN: GPIO.output(LED_YLW_PIN, GPIO.HIGH)
            if BUZZER_PIN:
                GPIO.output(BUZZER_PIN, GPIO.HIGH); time.sleep(0.05)
                GPIO.output(BUZZER_PIN, GPIO.LOW)
        elif level_color == "GREEN":
            if LED_GRN_PIN: GPIO.output(LED_GRN_PIN, GPIO.HIGH)
    except: pass


# ══════════════════════════════════════════════════════════════════════════════
# PERSONAL MODEL TRAINER (runs after calibration)
# ══════════════════════════════════════════════════════════════════════════════
class PersonalModelTrainer:
    def __init__(self, patient_name):
        self.patient_name  = patient_name
        self.model_file    = os.path.join(MODEL_PATH, f"{patient_name}_model.pkl")

    def train(self, rr_array, timestamps):
        """
        Train personal model from calibration phase RR data.
        rr_array:   numpy array of RR intervals in ms
        timestamps: numpy array of timestamps in seconds
        Returns: trained model dict
        """
        print(f"\n{'='*55}")
        print(f"TRAINING PERSONAL MODEL FOR: {self.patient_name}")
        print(f"{'='*55}")
        print(f"  Calibration beats: {len(rr_array)}")
        print(f"  Duration:          {timestamps[-1]/3600:.1f} hours")

        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import RobustScaler

        # Extract windows
        feats, labels, times_list = [], [], []
        for start in range(0, len(rr_array) - WINDOW_BEATS, STEP_BEATS):
            end    = start + WINDOW_BEATS
            rr_win = rr_array[start:end]
            if len(rr_win) < WINDOW_BEATS * 0.8: continue
            if np.std(rr_win) < 1e-6: continue
            feat = compute_features(rr_win)
            feats.append(feat)
            labels.append(0)   # all calibration = normal (no AF during calibration)
            times_list.append(float(timestamps[end-1]))

        if len(feats) < 50:
            print("  ❌  Not enough calibration windows. Continue recording.")
            return None

        X_calib = np.array(feats, dtype=np.float32)

        # Personal baseline
        mu  = X_calib.mean(0)
        std = X_calib.std(0) + 1e-8
        print(f"  Personal baseline computed from {len(feats)} windows")

        # Compute Kendall tau for phenotype
        from scipy.stats import kendalltau
        tau_vals = []
        try:
            tv, _ = kendalltau(np.arange(len(X_calib)), X_calib[:, 0])
            if np.isfinite(tv): tau_vals.append(float(tv))
        except: pass
        tau       = float(np.mean(tau_vals)) if tau_vals else 0.0
        phenotype = "rigid" if tau < -0.05 else ("unstable" if tau > 0.05 else "uncertain")
        print(f"  Kendall τ = {tau:.3f}  →  Phenotype: {phenotype.upper()}")

        # We don't have AF labels during calibration-only training.
        # Strategy: use the calibration windows as negatives, and generate
        # synthetic positives by perturbing features in the direction of
        # pre-AF change (based on phenotype).
        # This allows the model to distinguish 'this patient's normal' vs
        # 'deviated from normal' — the core of the personalised approach.

        X_norm = (X_calib - mu) / std

        # Generate synthetic pre-AF windows
        # Rigid phenotype:   variance ↓, regularity ↑, monotonicity ↑
        # Unstable phenotype: variance ↑, entropy ↑, autocorrelation ↑
        n_synth = max(int(len(X_norm) * 0.4), 100)
        rng     = np.random.RandomState(42)
        X_synth = X_norm[rng.choice(len(X_norm), n_synth, replace=True)].copy()

        if phenotype == "rigid":
            X_synth[:, 0]  -= rng.uniform(0.5, 2.0, n_synth)  # variance ↓
            X_synth[:, 18] += rng.uniform(0.5, 2.0, n_synth)  # regularity ↑
            X_synth[:, 19] += rng.uniform(0.3, 1.5, n_synth)  # monotonicity ↑
            X_synth[:, 20] -= rng.uniform(0.3, 1.5, n_synth)  # var stability ↓
        else:  # unstable or uncertain
            X_synth[:, 0]  += rng.uniform(0.5, 2.0, n_synth)  # variance ↑
            X_synth[:, 1]  += rng.uniform(0.3, 1.5, n_synth)  # autocorr ↑
            X_synth[:, 15] += rng.uniform(0.3, 1.5, n_synth)  # perm entropy ↑

        X_train = np.vstack([X_norm, X_synth])
        y_train = np.concatenate([np.zeros(len(X_norm)), np.ones(n_synth)])
        idx     = rng.permutation(len(X_train))
        X_train = X_train[idx]; y_train = y_train[idx]

        # Add trend features
        X_enh = add_trends(X_train)
        X_enh = np.nan_to_num(X_enh, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = RobustScaler()
        X_s    = scaler.fit_transform(X_enh)

        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=3, max_features='sqrt',
            random_state=42
        )
        clf.fit(X_s, y_train)

        model = {
            'clf': clf, 'scaler': scaler, 'mu': mu, 'std': std,
            'tau': tau, 'phenotype': phenotype,
            'patient': self.patient_name,
            'trained_at': datetime.now().isoformat(),
            'n_calib_windows': len(X_calib),
        }
        joblib.dump(model, self.model_file)
        print(f"\n  ✅  Model saved: {self.model_file}")
        print(f"  Phenotype: {phenotype.upper()}  |  τ = {tau:.3f}")
        print(f"  Ready for real-time prediction.\n")
        return model

    def load(self):
        if os.path.exists(self.model_file):
            return joblib.load(self.model_file)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# REAL-TIME MONITOR
# ══════════════════════════════════════════════════════════════════════════════
class RealtimeAFMonitor:
    def __init__(self, patient_name, model):
        self.patient_name  = patient_name
        self.clf           = model['clf']
        self.scaler        = model['scaler']
        self.mu            = model['mu']
        self.std           = model['std']
        self.tau           = model['tau']
        self.phenotype     = model['phenotype']

        self.rr_buffer     = deque(maxlen=WINDOW_BEATS * 10)  # rolling RR store
        self.feat_history  = deque(maxlen=TREND_WIN + 5)
        self.last_eval_t   = 0
        self.alert_history = []
        self.log_file      = os.path.join(LOG_PATH,
                                f"{patient_name}_{datetime.now():%Y%m%d_%H%M}.csv")

        # Write log header
        with open(self.log_file, 'w') as f:
            f.write("timestamp,rr_ms,risk_score,alert_level\n")

        setup_gpio_alerts()
        self._print_header()

    def _print_header(self):
        W = 55
        print(f"\n  ╔{'═'*W}╗")
        print(f"  ║{'  🫀  PERSONALISED AF MONITOR  ':^{W}}║")
        print(f"  ╠{'═'*W}╣")
        print(f"  ║  {'Patient: '+self.patient_name:<{W-4}}║")
        print(f"  ║  {'Phenotype: '+self.phenotype.upper()+' rhythm  |  τ='+str(round(self.tau,3)):<{W-4}}║")
        print(f"  ║  {'Alert interval: every '+str(EVAL_INTERVAL_S)+' seconds':<{W-4}}║")
        print(f"  ╚{'═'*W}╝\n")

    def add_rr(self, rr_ms, timestamp):
        """Add one new RR interval and evaluate if interval elapsed."""
        self.rr_buffer.append((rr_ms, timestamp))

        # Log raw RR
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp:.3f},{rr_ms:.1f},,\n")

        # Evaluate every EVAL_INTERVAL_S seconds
        if timestamp - self.last_eval_t >= EVAL_INTERVAL_S:
            self.last_eval_t = timestamp
            self._evaluate()

    def _evaluate(self):
        """Run the classifier on the current RR buffer and fire alert."""
        rrs = [r for r, _ in self.rr_buffer]
        if len(rrs) < WINDOW_BEATS:
            print(f"  ⏳  Collecting beats... ({len(rrs)}/{WINDOW_BEATS} needed)")
            return

        # Use most recent WINDOW_BEATS
        rr_win = np.array(rrs[-WINDOW_BEATS:], dtype=np.float64)

        # Feature extraction
        feat = compute_features(rr_win)
        self.feat_history.append(feat)

        if len(self.feat_history) < TREND_WIN + 1:
            print(f"  ⏳  Building trend history... ({len(self.feat_history)}/{TREND_WIN+1})")
            return

        # Normalise to personal baseline
        X_buf  = np.array(self.feat_history, dtype=np.float32)
        X_norm = (X_buf - self.mu) / self.std
        X_enh  = add_trends(X_norm)
        X_enh  = np.nan_to_num(X_enh, nan=0.0, posinf=0.0, neginf=0.0)

        feat_vec = X_enh[-1:].copy()
        feat_s   = self.scaler.transform(feat_vec)
        prob     = float(self.clf.predict_proba(feat_s)[0, 1])

        # Map to alert level
        alert_label, alert_desc, alert_color = "✅  STABLE", "NORMAL SINUS RHYTHM", "GREEN"
        for thresh, label, desc, col in ALERT_LEVELS:
            if prob >= thresh:
                alert_label, alert_desc, alert_color = label, desc, col
                break

        # Display
        ts_now   = datetime.now().strftime("%H:%M:%S")
        hr_now   = round(60000 / np.mean(rr_win[-10:]), 1)
        rr_now   = round(np.mean(rr_win[-5:]), 1)
        conf     = min(int(prob * 120), 99)

        W = 55
        print(f"\n  ╔{'═'*W}╗")
        print(f"  ║  {'Time: '+ts_now+'   HR: '+str(hr_now)+' bpm   RR: '+str(rr_now)+'ms':<{W-4}}║")
        print(f"  ╠{'─'*W}╣")
        print(f"  ║  {alert_label+' — '+alert_desc:<{W-4}}║")
        print(f"  ║  {'Risk score: '+str(round(prob,3))+'   Confidence: '+str(conf)+'%':<{W-4}}║")
        print(f"  ╚{'═'*W}╝")

        # GPIO alert
        fire_alert_gpio(alert_color)

        # Log alert
        with open(self.log_file, 'a') as f:
            f.write(f"{time.time():.3f},,{prob:.4f},{alert_label}\n")

        self.alert_history.append({
            'time': ts_now, 'prob': prob,
            'level': alert_label, 'hr': hr_now
        })


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION MODE (for testing without hardware)
# ══════════════════════════════════════════════════════════════════════════════
def simulate_from_file(patient_name, rr_file):
    """
    Test the monitor using saved LTAF RR data.
    Copy your .npz files to ~/af_monitor/recordings/
    """
    if not os.path.exists(rr_file):
        print(f"File not found: {rr_file}")
        print("Copy your .npz file to ~/af_monitor/recordings/")
        return

    print(f"Loading simulation data from: {rr_file}")
    d        = np.load(rr_file, allow_pickle=True)
    rr_ms    = d['rr_ms']
    rr_times = d['rr_times']

    trainer = PersonalModelTrainer(patient_name)
    model   = trainer.load()

    if model is None:
        print("No saved model found. Training from first 6h of data...")
        calib_mask = rr_times <= 6 * 3600
        model      = trainer.train(rr_ms[calib_mask], rr_times[calib_mask])
        if model is None:
            print("Training failed."); return

    monitor = RealtimeAFMonitor(patient_name, model)

    # Feed RR intervals at accelerated speed (10x) for demo
    calib_end = 6 * 3600
    pred_mask = rr_times > calib_end
    rr_pred   = rr_ms[pred_mask]
    t_pred    = rr_times[pred_mask]

    print(f"\nRunning prediction on {len(rr_pred)} beats ({len(rr_pred)/3600*0.85:.1f}h recording)...")
    print("Press Ctrl+C to stop\n")

    try:
        for rr, t in zip(rr_pred, t_pred):
            monitor.add_rr(float(rr), float(t))
            time.sleep(0.002)   # 10x speed
    except KeyboardInterrupt:
        print("\nStopped.")
    print(f"\nLog saved: {monitor.log_file}")


# ══════════════════════════════════════════════════════════════════════════════
# LIVE HARDWARE MODE
# ══════════════════════════════════════════════════════════════════════════════
def run_live(patient_name, mode):
    """
    Live mode — reads from AD8232 via MCP3008 SPI.
    mode: 'calibrate' | 'predict' | 'auto'
    """
    if not HARDWARE_AVAILABLE:
        print("Hardware not available. Use --mode simulate instead.")
        return

    GPIO.setmode(GPIO.BCM)
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 1350000

    trainer  = PersonalModelTrainer(patient_name)
    detector = RPeakDetector(fs=SAMPLE_RATE)

    rr_calib   = []
    t_calib    = []
    start_time = time.time()
    sample_dt  = 1.0 / SAMPLE_RATE
    beat_count = 0

    print(f"\n{'='*55}")
    print(f"  AF MONITOR — LIVE MODE: {mode.upper()}")
    print(f"  Patient: {patient_name}")
    print(f"{'='*55}\n")

    # ── CALIBRATION PHASE ────────────────────────────────────────────────────
    if mode in ('calibrate', 'auto'):
        calib_duration = CALIBRATION_H * 3600
        print(f"  CALIBRATION PHASE ({CALIBRATION_H:.0f} hours)")
        print(f"  Collecting your personal RR baseline...")
        print(f"  Sit quietly, breathe normally. No AF should occur during calibration.\n")

        while True:
            elapsed = time.time() - start_time
            if elapsed >= calib_duration and len(rr_calib) >= MIN_CALIB_BEATS:
                break

            t0     = time.time()
            if not check_leads():
                print(f"  ⚠  Lead off — check electrodes"); time.sleep(1); continue

            sample = read_adc(ADC_CHANNEL)
            rr     = detector.process(sample)

            if rr is not None:
                beat_count += 1
                rr_calib.append(rr)
                t_calib.append(elapsed)

                if beat_count % 300 == 0:   # print every 5 minutes
                    hr = 60000 / rr
                    pct = min(elapsed / calib_duration * 100, 100)
                    print(f"  [{pct:4.0f}%]  Beat {beat_count}  HR={hr:.0f}bpm  "
                          f"Time={elapsed/3600:.2f}h")

            elapsed = time.time() - t0
            sleep   = sample_dt - elapsed
            if sleep > 0: time.sleep(sleep)

        print(f"\n  Calibration complete. {len(rr_calib)} beats collected.")
        model = trainer.train(np.array(rr_calib), np.array(t_calib))

        if model is None:
            print("Training failed."); spi.close(); GPIO.cleanup(); return

        if mode == 'calibrate':
            print("  Calibration done. Run with --mode predict to start monitoring.")
            spi.close(); GPIO.cleanup(); return

    # ── PREDICTION PHASE ─────────────────────────────────────────────────────
    model = trainer.load()
    if model is None:
        print("No model found. Run --mode calibrate first.")
        spi.close(); GPIO.cleanup(); return

    monitor   = RealtimeAFMonitor(patient_name, model)
    pred_start = time.time()

    print(f"  PREDICTION PHASE — monitoring in real time")
    print(f"  Press Ctrl+C to stop\n")

    try:
        while True:
            t0      = time.time()
            elapsed = time.time() - pred_start

            if not check_leads():
                print(f"  ⚠  Lead off — reattach electrodes"); time.sleep(1); continue

            sample = read_adc(ADC_CHANNEL)
            rr     = detector.process(sample)

            if rr is not None:
                monitor.add_rr(rr, elapsed)

            sleep = sample_dt - (time.time() - t0)
            if sleep > 0: time.sleep(sleep)

    except KeyboardInterrupt:
        print("\n  Monitoring stopped.")
    finally:
        spi.close()
        GPIO.cleanup()
        print(f"  Log saved: {monitor.log_file}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Personalised AF Edge Monitor')
    parser.add_argument('--mode',    default='auto',
                        choices=['calibrate','predict','auto','simulate'],
                        help='Operating mode')
    parser.add_argument('--patient', default='patient01',
                        help='Patient identifier (used for saving model)')
    parser.add_argument('--file',    default=None,
                        help='Path to .npz RR file for simulation mode')
    args = parser.parse_args()

    if args.mode == 'simulate' or not HARDWARE_AVAILABLE:
        # Simulation using saved LTAF data
        rr_file = args.file or os.path.join(
            os.path.expanduser("~"), "af_monitor", "recordings",
            f"{args.patient}_rr.npz"
        )
        simulate_from_file(args.patient, rr_file)
    else:
        run_live(args.patient, args.mode)
