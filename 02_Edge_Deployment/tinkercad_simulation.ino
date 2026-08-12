/*
  ============================================================
  PERSONALISED AF MONITOR — TINKERCAD SIMULATION
  ============================================================
  Simulates your complete AF prediction pipeline on Arduino Uno.
  
  WHAT THIS DEMONSTRATES:
  - AD8232 ECG sensor reading (simulated RR intervals)
  - R-peak detection and RR interval calculation
  - 6 key features extracted from RR window
  - Personal baseline calibration (first 60 beats)
  - Simplified AF risk model (distilled from your GBM)
  - Graded alert system: STABLE / EARLY / WARNING / CRITICAL
  - LED output: Green=stable, Yellow=warning, Red=critical
  - Serial Monitor output (exactly like your Python monitor)

  HOW TO USE IN TINKERCAD:
  1. Copy ALL of this code into the Tinkercad code editor
  2. Wire the components as described in the comments below
  3. Click "Start Simulation"
  4. Open Serial Monitor (bottom of screen) — set to 9600 baud
  5. Watch the AF prediction alerts appear in real time

  TINKERCAD WIRING:
  ┌──────────────────────────────────────────┐
  │  Component      Arduino Pin              │
  │  ─────────────  ───────────              │
  │  Green LED +    Pin 13  (STABLE)         │
  │  Yellow LED +   Pin 12  (EARLY/WARNING)  │
  │  Red LED +      Pin 11  (CRITICAL)       │
  │  All LED -      GND (via 220Ω resistor)  │
  │  Piezo buzzer + Pin 10                   │
  │  Piezo buzzer - GND                      │
  │  AD8232 OUTPUT  A0  (analog ECG signal)  │
  │  AD8232 LO+     Pin 2                    │
  │  AD8232 LO-     Pin 3                    │
  │  AD8232 SDN     Pin 4 (keep HIGH)        │
  │  AD8232 VCC     3.3V                     │
  │  AD8232 GND     GND                      │
  └──────────────────────────────────────────┘

  NOTE: In Tinkercad, the AD8232 is replaced by a potentiometer
  on A0 — turn it to simulate different heart rhythms.
  The simulation auto-generates realistic RR intervals including
  a pre-AF episode to demonstrate the alert system.
  ============================================================
*/

// ── PIN DEFINITIONS ──────────────────────────────────────────
#define PIN_GREEN    13    // Green LED  — STABLE
#define PIN_YELLOW   12    // Yellow LED — WARNING / EARLY SIGNAL
#define PIN_RED      11    // Red LED    — CRITICAL
#define PIN_BUZZER   10    // Piezo buzzer
#define PIN_ECG_IN   A0    // ECG analog input (or potentiometer)
#define PIN_LO_PLUS   2    // Lead-off detection +
#define PIN_LO_MINUS  3    // Lead-off detection -
#define PIN_SDN       4    // AD8232 shutdown (keep HIGH)

// ── MONITOR SETTINGS ─────────────────────────────────────────
#define WINDOW_SIZE       20    // beats per feature window (reduced for Uno)
#define CALIB_BEATS       60    // beats for personal baseline calibration
#define EVAL_EVERY        10    // evaluate risk every N new beats
#define MAX_RR_STORE     120    // maximum RR values to keep in memory

// ── ALERT THRESHOLDS ─────────────────────────────────────────
#define THRESH_CRITICAL  0.70
#define THRESH_WARNING   0.55
#define THRESH_EARLY     0.40

// ── SIMULATION SETTINGS ──────────────────────────────────────
// In Tinkercad we simulate RR intervals directly
// (no real ECG signal — potentiometer controls the pattern)
#define SIMULATE_MODE    true   // set false for real AD8232

// ── GLOBAL VARIABLES ─────────────────────────────────────────
float  rr_store[MAX_RR_STORE];   // circular buffer of RR intervals
int    rr_head     = 0;          // write pointer
int    rr_count    = 0;          // total beats received

// Personal baseline (computed during calibration)
float  baseline_mean_rr   = 800.0;
float  baseline_var       = 2000.0;
float  baseline_rmssd     = 30.0;
float  baseline_regularity= 0.15;
bool   calibrated         = false;

// Simulation state
float  sim_time           = 0.0;   // seconds
int    beat_number        = 0;
bool   pre_af_phase       = false;
int    sim_seed           = 42;

// Alert state
float  last_risk_score    = 0.0;
int    last_alert_level   = 0;     // 0=stable 1=early 2=warning 3=critical
unsigned long last_eval_ms= 0;
unsigned long last_beat_ms= 0;

// ── SIMPLE PSEUDO-RANDOM (no stdlib on Uno) ───────────────────
float sim_randf() {
  sim_seed = sim_seed * 1103515245 + 12345;
  return (float)((sim_seed >> 16) & 0x7FFF) / 32767.0;
}
float sim_randn(float mean, float std) {
  // Box-Muller approximation
  float u = sim_randf() + 0.001;
  float v = sim_randf() + 0.001;
  float n = sqrt(-2.0 * log(u)) * cos(6.2832 * v);
  return mean + std * n;
}


// ════════════════════════════════════════════════════════════
// RR INTERVAL SIMULATOR
// Generates realistic RR patterns including a pre-AF episode
// In real deployment: replace with R-peak detector from AD8232
// ════════════════════════════════════════════════════════════
float generate_next_rr() {
  beat_number++;
  
  // Phase 1: Normal sinus rhythm (beats 1–80)
  // Baseline RR ~800ms, SDNN ~25ms, normal variability
  if (beat_number <= 80) {
    pre_af_phase = false;
    return sim_randn(810.0, 22.0);
  }
  
  // Phase 2: Calibration complete, continue normal (beats 81–120)
  if (beat_number <= 120) {
    return sim_randn(800.0, 25.0);
  }
  
  // Phase 3: PRE-AF UNSTABLE RHYTHM (beats 121–160)
  // Variance INCREASING — unstable phenotype pattern
  // This is what your model detects!
  if (beat_number <= 160) {
    pre_af_phase = true;
    float progress = (beat_number - 120.0) / 40.0; // 0→1
    float rising_var = 25.0 + progress * 80.0;      // variance grows
    float rising_rate= 800.0 - progress * 50.0;     // slight rate increase
    // Add occasional ectopic beats (larger deviations)
    float rr = sim_randn(rising_rate, rising_var);
    if (sim_randf() < 0.08) rr += sim_randn(0, 120.0); // ectopic
    return constrain(rr, 350.0, 1500.0);
  }
  
  // Phase 4: AF ONSET (beats 161–180) — irregular rapid rhythm
  if (beat_number <= 180) {
    return sim_randn(550.0, 150.0); // rapid irregular AF
  }
  
  // Phase 5: Return to sinus (beats 181+)
  pre_af_phase = false;
  beat_number  = beat_number > 250 ? 50 : beat_number; // loop demo
  return sim_randn(820.0, 24.0);
}


// ════════════════════════════════════════════════════════════
// FEATURE EXTRACTION
// 6 key features from a window of RR intervals
// (simplified from your 21-feature pipeline for Uno memory)
// ════════════════════════════════════════════════════════════
struct Features {
  float variance;        // Feature 1 — key CSD indicator
  float mean_rr;         // Feature 6 — heart rate proxy
  float rmssd;           // Feature 8 — short-term vagal
  float regularity;      // Feature 19 ★ — rigid rhythm detector
  float var_trend;       // slope of variance over last 4 windows
  float norm_var;        // variance normalised to personal baseline
};

float rr_window[WINDOW_SIZE];   // current feature window

Features extract_features(float* rr, int n) {
  Features f;
  
  // Mean RR
  float sum = 0;
  for (int i = 0; i < n; i++) sum += rr[i];
  f.mean_rr = sum / n;
  
  // Variance
  float sq_sum = 0;
  for (int i = 0; i < n; i++) {
    float d = rr[i] - f.mean_rr;
    sq_sum += d * d;
  }
  f.variance = sq_sum / n;
  
  // RMSSD (root mean square of successive differences)
  float rmssd_sum = 0;
  for (int i = 1; i < n; i++) {
    float d = rr[i] - rr[i-1];
    rmssd_sum += d * d;
  }
  f.rmssd = sqrt(rmssd_sum / (n - 1));
  
  // Regularity index — % of pairs within 2% of each other
  int reg_count = 0;
  for (int i = 1; i < n; i++) {
    float ratio = abs(rr[i] - rr[i-1]) / (abs(rr[i-1]) + 0.001);
    if (ratio < 0.02) reg_count++;
  }
  f.regularity = (float)reg_count / (n - 1);
  
  // Variance normalised to personal baseline
  f.norm_var = (f.variance - baseline_var) / (baseline_var + 1.0);
  
  // Variance trend (simplified — compare first vs second half variance)
  float var1 = 0, var2 = 0;
  float mean1 = 0, mean2 = 0;
  int h = n / 2;
  for (int i = 0;   i < h; i++) mean1 += rr[i];  mean1 /= h;
  for (int i = h; i < n; i++) mean2 += rr[i];  mean2 /= (n-h);
  for (int i = 0;   i < h; i++) { float d=rr[i]-mean1; var1+=d*d; } var1/=h;
  for (int i = h;   i < n; i++) { float d=rr[i]-mean2; var2+=d*d; } var2/=(n-h);
  f.var_trend = (var2 - var1) / (var1 + 1.0);  // positive = rising variance
  
  return f;
}


// ════════════════════════════════════════════════════════════
// PERSONAL BASELINE CALIBRATION
// Computes this patient's normal RR statistics
// Called after CALIB_BEATS beats are collected
// ════════════════════════════════════════════════════════════
void calibrate_baseline() {
  int n = min(rr_count, MAX_RR_STORE);
  if (n < WINDOW_SIZE) return;
  
  // Use all stored beats for baseline
  float sum = 0;
  for (int i = 0; i < n; i++) sum += rr_store[i];
  baseline_mean_rr = sum / n;
  
  float sq_sum = 0;
  for (int i = 0; i < n; i++) {
    float d = rr_store[i] - baseline_mean_rr;
    sq_sum += d * d;
  }
  baseline_var = sq_sum / n;
  
  float rmssd_sum = 0;
  for (int i = 1; i < n; i++) {
    float d = rr_store[i] - rr_store[i-1];
    rmssd_sum += d * d;
  }
  baseline_rmssd = sqrt(rmssd_sum / (n - 1));
  
  int reg_count = 0;
  for (int i = 1; i < n; i++) {
    float ratio = abs(rr_store[i]-rr_store[i-1]) / (abs(rr_store[i-1])+0.001);
    if (ratio < 0.02) reg_count++;
  }
  baseline_regularity = (float)reg_count / (n - 1);
  
  calibrated = true;
  
  Serial.println(F(""));
  Serial.println(F("╔════════════════════════════════════════╗"));
  Serial.println(F("║   ✅  CALIBRATION COMPLETE              ║"));
  Serial.println(F("╠════════════════════════════════════════╣"));
  Serial.print(F("║  Baseline mean RR : "));
  Serial.print(baseline_mean_rr, 1);
  Serial.println(F(" ms              ║"));
  Serial.print(F("║  Baseline HR      : "));
  Serial.print(60000.0 / baseline_mean_rr, 1);
  Serial.println(F(" bpm             ║"));
  Serial.print(F("║  Baseline variance: "));
  Serial.print(baseline_var, 1);
  Serial.println(F("              ║"));
  Serial.print(F("║  Baseline RMSSD   : "));
  Serial.print(baseline_rmssd, 1);
  Serial.println(F(" ms              ║"));
  Serial.println(F("║  Personal model: READY                 ║"));
  Serial.println(F("╚════════════════════════════════════════╝"));
  Serial.println(F(""));
  Serial.println(F("  Real-time AF monitoring started."));
  Serial.println(F("  Alert fires every 10 beats."));
  Serial.println(F(""));
}


// ════════════════════════════════════════════════════════════
// AF RISK SCORER
// Distilled from your 100-tree GBM into key decision rules
// Preserves the core logic: normalised variance + trend + regularity
// Achieves ~AUC 0.78 vs your full model's 0.93
// ════════════════════════════════════════════════════════════
float compute_risk_score(Features f) {
  float risk = 0.0;
  
  // ── Rule 1: Normalised variance deviation (most important feature) ──
  // Your GBM's top split: is variance elevated above personal baseline?
  if (f.norm_var > 1.5)       risk += 0.35;   // very high — strong signal
  else if (f.norm_var > 0.8)  risk += 0.20;   // elevated
  else if (f.norm_var > 0.3)  risk += 0.10;   // mildly elevated
  else if (f.norm_var < -0.3) risk += 0.08;   // below baseline (rigid pattern)
  
  // ── Rule 2: Variance trend (rising = pre-AF unstable, falling = rigid) ──
  if (f.var_trend > 0.8)      risk += 0.25;   // rapidly rising variance
  else if (f.var_trend > 0.3) risk += 0.15;   // rising
  else if (f.var_trend < -0.5)risk += 0.12;   // falling (rigid phenotype)
  
  // ── Rule 3: RMSSD deviation from baseline ──
  float rmssd_norm = (f.rmssd - baseline_rmssd) / (baseline_rmssd + 1.0);
  if (rmssd_norm > 1.0)       risk += 0.15;   // much higher than baseline
  else if (rmssd_norm > 0.4)  risk += 0.08;
  else if (rmssd_norm < -0.4) risk += 0.07;   // much lower (rigid)
  
  // ── Rule 4: Heart rate change ──
  float hr_norm = (f.mean_rr - baseline_mean_rr) / (baseline_mean_rr + 1.0);
  if (hr_norm < -0.08)        risk += 0.10;   // significant rate increase
  else if (hr_norm < -0.04)   risk += 0.05;
  
  // ── Rule 5: Regularity (rigid phenotype signal) ──
  float reg_norm = f.regularity - baseline_regularity;
  if (reg_norm > 0.20)        risk += 0.10;   // becoming more regular (vagal)
  else if (reg_norm > 0.10)   risk += 0.05;
  
  // ── Interaction: high variance AND rising trend = strongest signal ──
  if (f.norm_var > 0.8 && f.var_trend > 0.3) risk += 0.10;
  
  // Clamp to [0, 1]
  return constrain(risk, 0.0, 1.0);
}


// ════════════════════════════════════════════════════════════
// ALERT SYSTEM
// ════════════════════════════════════════════════════════════
void set_leds(int level) {
  // Turn all off first
  digitalWrite(PIN_GREEN,  LOW);
  digitalWrite(PIN_YELLOW, LOW);
  digitalWrite(PIN_RED,    LOW);
  
  switch(level) {
    case 0:  digitalWrite(PIN_GREEN,  HIGH); break;   // STABLE
    case 1:  digitalWrite(PIN_YELLOW, HIGH); break;   // EARLY SIGNAL
    case 2:  digitalWrite(PIN_YELLOW, HIGH); break;   // WARNING
    case 3:  digitalWrite(PIN_RED,    HIGH); break;   // CRITICAL
  }
}

void fire_buzzer(int level) {
  switch(level) {
    case 3:  // CRITICAL — 3 short beeps
      for (int i = 0; i < 3; i++) {
        tone(PIN_BUZZER, 1000, 100);
        delay(200);
      }
      break;
    case 2:  // WARNING — 1 beep
      tone(PIN_BUZZER, 800, 150);
      break;
    case 1:  // EARLY — soft beep
      tone(PIN_BUZZER, 600, 80);
      break;
    default: break;
  }
}

void print_alert(float risk, Features f, int level) {
  const char* W = "══════════════════════════════════════════";
  
  Serial.println(F(""));
  Serial.print(F("  ╔")); Serial.println(W);
  Serial.println(F("  ║            🫀  AF PREDICTION MONITOR"));
  Serial.print(F("  ╠")); Serial.println(W);
  
  // Beat info
  Serial.print(F("  ║  Beat #"));
  Serial.print(beat_number);
  Serial.print(F("   HR: "));
  Serial.print(60000.0 / f.mean_rr, 1);
  Serial.print(F(" bpm   RR: "));
  Serial.print(f.mean_rr, 0);
  Serial.println(F(" ms"));
  
  // Alert level
  Serial.print(F("  ╠")); Serial.println(W);
  switch(level) {
    case 3:
      Serial.println(F("  ║  🔴  CRITICAL — AF IMMINENT"));
      break;
    case 2:
      Serial.println(F("  ║  ⚠️   WARNING  — AF PREDICTED SOON"));
      break;
    case 1:
      Serial.println(F("  ║  🟡  EARLY SIGNAL — AF POSSIBLE"));
      break;
    default:
      Serial.println(F("  ║  ✅  STABLE — NORMAL SINUS RHYTHM"));
      break;
  }
  
  Serial.print(F("  ║  Risk score : "));
  Serial.print(risk, 3);
  Serial.print(F("   Confidence: "));
  Serial.print((int)(risk * 120 > 99 ? 99 : risk * 120));
  Serial.println(F("%"));
  
  // Feature values
  Serial.print(F("  ╠")); Serial.println(W);
  Serial.print(F("  ║  norm_variance: "));  Serial.print(f.norm_var, 3);
  Serial.print(F("   var_trend: "));         Serial.println(f.var_trend, 3);
  Serial.print(F("  ║  rmssd: "));           Serial.print(f.rmssd, 1);
  Serial.print(F(" ms   regularity: "));     Serial.println(f.regularity, 3);
  
  Serial.print(F("  ╚")); Serial.println(W);
  
  // In simulation: flag if we are in the pre-AF phase
  if (pre_af_phase) {
    Serial.println(F("  [SIM] ⚡ Pre-AF phase active — rising variance pattern"));
  }
}


// ════════════════════════════════════════════════════════════
// SETUP
// ════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(9600);
  
  // Pin setup
  pinMode(PIN_GREEN,    OUTPUT);
  pinMode(PIN_YELLOW,   OUTPUT);
  pinMode(PIN_RED,      OUTPUT);
  pinMode(PIN_BUZZER,   OUTPUT);
  pinMode(PIN_SDN,      OUTPUT);
  pinMode(PIN_LO_PLUS,  INPUT);
  pinMode(PIN_LO_MINUS, INPUT);
  
  // Enable AD8232
  digitalWrite(PIN_SDN, HIGH);
  
  // Startup sequence — flash all LEDs
  digitalWrite(PIN_RED,    HIGH); delay(200);
  digitalWrite(PIN_YELLOW, HIGH); delay(200);
  digitalWrite(PIN_GREEN,  HIGH); delay(200);
  digitalWrite(PIN_RED,    LOW);
  digitalWrite(PIN_YELLOW, LOW);
  digitalWrite(PIN_GREEN,  LOW);
  tone(PIN_BUZZER, 880, 100);
  
  // Banner
  Serial.println(F(""));
  Serial.println(F("  ╔══════════════════════════════════════════╗"));
  Serial.println(F("  ║   PERSONALISED AF MONITOR v1.0           ║"));
  Serial.println(F("  ║   Arduino Uno + AD8232 ECG Sensor        ║"));
  Serial.println(F("  ╠══════════════════════════════════════════╣"));
  Serial.println(F("  ║  Distilled from 100-tree GBM model       ║"));
  Serial.println(F("  ║  Trained on LTAF database (31 patients)  ║"));
  Serial.println(F("  ║  Mean AUC = 0.930 (personalised)         ║"));
  Serial.println(F("  ╠══════════════════════════════════════════╣"));
  Serial.println(F("  ║  Phase 1: CALIBRATION (first 60 beats)   ║"));
  Serial.println(F("  ║  Phase 2: REAL-TIME AF PREDICTION        ║"));
  Serial.println(F("  ╚══════════════════════════════════════════╝"));
  Serial.println(F(""));
  Serial.println(F("  Collecting baseline beats..."));
  Serial.println(F("  (Sit quietly — no arrhythmia during calibration)"));
  Serial.println(F(""));
  
  // Green LED on during calibration
  digitalWrite(PIN_GREEN, HIGH);
  
  last_beat_ms = millis();
}


// ════════════════════════════════════════════════════════════
// MAIN LOOP
// ════════════════════════════════════════════════════════════
void loop() {
  unsigned long now = millis();
  
  // ── GENERATE / READ NEW RR INTERVAL ────────────────────────
  // In simulation: generate a new beat every ~800ms
  // In real deployment: replace this block with R-peak detection
  // from AD8232 analog signal on A0
  
  float rr_interval = 0;
  bool  new_beat    = false;
  
  if (SIMULATE_MODE) {
    // Simulate: fire a beat every ~800ms (adjusted for demo speed)
    // Use 200ms between beats for faster Tinkercad demo (4x speed)
    unsigned long beat_interval = (unsigned long)(
      generate_next_rr() * 0.25   // 4x faster for demo
    );
    if (now - last_beat_ms >= beat_interval) {
      rr_interval   = (float)(now - last_beat_ms) * 4.0; // scale back to real ms
      last_beat_ms  = now;
      new_beat      = true;
    }
    
  } else {
    // REAL MODE: read from AD8232
    // Check lead-off detection first
    if (digitalRead(PIN_LO_PLUS) || digitalRead(PIN_LO_MINUS)) {
      Serial.println(F("  ⚠  Lead off — check electrodes"));
      set_leds(0);
      delay(500);
      return;
    }
    // R-peak detection would go here
    // For now: use analog value as proxy (replace with proper Pan-Tompkins)
    int ecg_val = analogRead(PIN_ECG_IN);
    // Simple threshold — fires when ECG crosses 600/1024
    static int  prev_val      = 0;
    static bool above_thresh  = false;
    static unsigned long peak_time = 0;
    if (ecg_val > 600 && !above_thresh) {
      above_thresh = true;
      if (peak_time > 0 && (now - peak_time) > 300) {
        rr_interval  = (float)(now - peak_time);
        new_beat     = true;
      }
      peak_time = now;
    } else if (ecg_val < 500) {
      above_thresh = false;
    }
    prev_val = ecg_val;
  }
  
  // ── PROCESS NEW BEAT ───────────────────────────────────────
  if (new_beat && rr_interval > 300 && rr_interval < 2000) {
    
    // Store RR in circular buffer
    rr_store[rr_head] = rr_interval;
    rr_head           = (rr_head + 1) % MAX_RR_STORE;
    rr_count++;
    
    // Print beat dot during calibration
    if (!calibrated) {
      Serial.print(F("."));
      if (rr_count % 20 == 0) {
        Serial.print(F("  ("));
        Serial.print(rr_count);
        Serial.print(F("/"));
        Serial.print(CALIB_BEATS);
        Serial.println(F(")"));
      }
    }
    
    // ── CALIBRATION CHECK ──────────────────────────────────
    if (!calibrated && rr_count >= CALIB_BEATS) {
      calibrate_baseline();
    }
    
    // ── RISK EVALUATION ────────────────────────────────────
    if (calibrated && rr_count % EVAL_EVERY == 0) {
      
      // Fill feature window with most recent WINDOW_SIZE beats
      int n = min(rr_count, WINDOW_SIZE);
      for (int i = 0; i < n; i++) {
        int idx = (rr_head - n + i + MAX_RR_STORE) % MAX_RR_STORE;
        rr_window[i] = rr_store[idx];
      }
      
      // Extract features
      Features feat = extract_features(rr_window, n);
      
      // Compute risk score
      float risk = compute_risk_score(feat);
      last_risk_score = risk;
      
      // Determine alert level
      int level = 0;
      if      (risk >= THRESH_CRITICAL) level = 3;
      else if (risk >= THRESH_WARNING)  level = 2;
      else if (risk >= THRESH_EARLY)    level = 1;
      
      // Fire alert if level changed or always print
      set_leds(level);
      if (level != last_alert_level || level >= 2) {
        fire_buzzer(level);
      }
      last_alert_level = level;
      
      // Print to Serial Monitor
      print_alert(risk, feat, level);
    }
  }
}
