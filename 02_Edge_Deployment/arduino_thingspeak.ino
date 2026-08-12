/*
  ============================================================
  PERSONALISED AF MONITOR — ThingSpeak via USB Serial
  ============================================================
  Hardware:  Arduino Uno + AD8232 ECG sensor
  Connection: USB cable to PC
  Cloud:     ThingSpeak (via Python bridge on PC)

  WIRING:
    AD8232 OUTPUT  →  Arduino A0
    AD8232 LO+     →  Arduino Pin 2
    AD8232 LO-     →  Arduino Pin 3
    AD8232 SDN     →  Arduino Pin 4
    AD8232 3.3V    →  Arduino 3.3V
    AD8232 GND     →  Arduino GND

    Green  LED + 220Ω  →  Arduino Pin 13  (STABLE)
    Yellow LED + 220Ω  →  Arduino Pin 12  (WARNING)
    Red    LED + 220Ω  →  Arduino Pin 11  (CRITICAL)
    Piezo buzzer       →  Arduino Pin 10

  THINGSPEAK CHANNEL FIELDS:
    Field 1 → RR interval (ms)
    Field 2 → Heart rate (bpm)
    Field 3 → Normalised variance
    Field 4 → RMSSD (ms)
    Field 5 → Risk score (0.0 – 1.0)
    Field 6 → Alert level (0=stable 1=early 2=warning 3=critical)
    Field 7 → Regularity index
    Field 8 → Variance trend

  HOW TO USE:
    1. Upload this sketch to Arduino Uno
    2. Keep USB cable connected to PC
    3. Run af_thingspeak_bridge.py on PC
    4. Open ThingSpeak dashboard — data appears every 15 seconds
  ============================================================
*/

// ── PINS ─────────────────────────────────────────────────────
#define PIN_ECG_IN    A0
#define PIN_LO_PLUS    2
#define PIN_LO_MINUS   3
#define PIN_SDN        4
#define PIN_RED       11
#define PIN_YELLOW    12
#define PIN_GREEN     13
#define PIN_BUZZER    10

// ── SETTINGS ─────────────────────────────────────────────────
#define WINDOW_SIZE      20    // beats per feature window
#define CALIB_BEATS      60    // calibration phase length
#define EVAL_EVERY       10    // evaluate every N beats
#define MAX_RR          120    // circular RR buffer size
#define SEND_EVERY_MS  15000   // send to ThingSpeak every 15s
                               // (ThingSpeak free = 15s minimum)

// ── SERIAL PROTOCOL ──────────────────────────────────────────
// Arduino prints one line every SEND_EVERY_MS milliseconds:
// "DATA,<rr>,<hr>,<norm_var>,<rmssd>,<risk>,<level>,<reg>,<var_trend>"
// Python bridge reads this line and POSTs to ThingSpeak.
// All other Serial output is prefixed with "#" (ignored by bridge).
#define DATA_PREFIX  "DATA,"
#define LOG_PREFIX   "# "

// ── GLOBAL STATE ─────────────────────────────────────────────
float  rr_buf[MAX_RR];
int    rr_head    = 0;
int    rr_count   = 0;

// Personal baseline (set during calibration)
float  bl_mean    = 800.0;
float  bl_var     = 500.0;
float  bl_rmssd   = 28.0;
float  bl_reg     = 0.15;
bool   calibrated = false;

// R-peak detection state
int    ecg_prev      = 0;
bool   above_thresh  = false;
unsigned long peak_ms    = 0;
unsigned long last_rr_ms = 0;

// Output state
float  last_rr    = 0;
float  last_hr    = 0;
float  last_nvar  = 0;
float  last_rmssd = 0;
float  last_risk  = 0;
int    last_level = 0;
float  last_reg   = 0;
float  last_vtrend= 0;
unsigned long last_send_ms = 0;

// ── FEATURE STRUCT ────────────────────────────────────────────
struct Feat {
  float mean_rr;
  float variance;
  float rmssd;
  float regularity;
  float norm_var;
  float var_trend;
};

// ── PSEUDO-RANDOM (for simulation fallback) ───────────────────
long  _seed = 12345;
float rng() {
  _seed = _seed * 1103515245L + 12345L;
  return (float)((_seed >> 16) & 0x7FFF) / 32767.0;
}
float rng_normal(float mu, float sd) {
  float u = rng() + 0.001, v = rng() + 0.001;
  return mu + sd * sqrt(-2.0 * log(u)) * cos(6.2832 * v);
}

// ── SIMULATION (set true if no AD8232 connected) ─────────────
#define SIM_MODE  false   // ← change to true to test without sensor

int    sim_beat   = 0;
unsigned long sim_last = 0;

float sim_next_rr() {
  sim_beat++;
  if (sim_beat <= 80)  return rng_normal(810, 22);   // normal
  if (sim_beat <= 120) return rng_normal(800, 25);   // normal
  if (sim_beat <= 160) {                              // pre-AF
    float p = (sim_beat - 120.0) / 40.0;
    return constrain(rng_normal(800 - p*50, 25 + p*80), 350, 1500);
  }
  if (sim_beat <= 180) return rng_normal(550, 150);  // AF
  if (sim_beat > 240)  sim_beat = 50;
  return rng_normal(815, 24);                        // recovery
}


// ════════════════════════════════════════════════════════════
// FEATURE EXTRACTION (6 features — memory-safe for 2KB RAM)
// ════════════════════════════════════════════════════════════
Feat extract(float* rr, int n) {
  Feat f;

  // Mean RR
  float s = 0;
  for (int i = 0; i < n; i++) s += rr[i];
  f.mean_rr = s / n;

  // Variance
  float sq = 0;
  for (int i = 0; i < n; i++) {
    float d = rr[i] - f.mean_rr; sq += d * d;
  }
  f.variance = sq / n;

  // RMSSD
  float rs = 0;
  for (int i = 1; i < n; i++) {
    float d = rr[i] - rr[i-1]; rs += d * d;
  }
  f.rmssd = sqrt(rs / max(n - 1, 1));

  // Regularity index (% pairs within 2% of each other)
  int rc = 0;
  for (int i = 1; i < n; i++) {
    if (abs(rr[i] - rr[i-1]) / (abs(rr[i-1]) + 0.001) < 0.02) rc++;
  }
  f.regularity = (float)rc / max(n - 1, 1);

  // Normalised variance (deviation from personal baseline)
  f.norm_var = (f.variance - bl_var) / (bl_var + 1.0);

  // Variance trend (2nd half vs 1st half variance)
  int h = n / 2;
  float m1=0, m2=0, v1=0, v2=0;
  for (int i=0;   i<h; i++) m1 += rr[i];   m1 /= max(h,1);
  for (int i=h;   i<n; i++) m2 += rr[i];   m2 /= max(n-h,1);
  for (int i=0;   i<h; i++) { float d=rr[i]-m1; v1+=d*d; } v1/=max(h,1);
  for (int i=h;   i<n; i++) { float d=rr[i]-m2; v2+=d*d; } v2/=max(n-h,1);
  f.var_trend = (v2 - v1) / (v1 + 1.0);

  return f;
}


// ════════════════════════════════════════════════════════════
// RISK SCORER (distilled from 100-tree GBM)
// Core decision rules extracted from top feature splits
// ════════════════════════════════════════════════════════════
float score(Feat f) {
  float r = 0.0;

  // Rule 1: Normalised variance (top GBM feature)
  if      (f.norm_var > 1.5)  r += 0.35;
  else if (f.norm_var > 0.8)  r += 0.20;
  else if (f.norm_var > 0.3)  r += 0.10;
  else if (f.norm_var < -0.3) r += 0.08;   // rigid phenotype

  // Rule 2: Variance trend
  if      (f.var_trend > 0.8)  r += 0.25;
  else if (f.var_trend > 0.3)  r += 0.15;
  else if (f.var_trend < -0.5) r += 0.12;  // rigid phenotype

  // Rule 3: RMSSD deviation
  float rn = (f.rmssd - bl_rmssd) / (bl_rmssd + 1.0);
  if      (rn >  1.0) r += 0.15;
  else if (rn >  0.4) r += 0.08;
  else if (rn < -0.4) r += 0.07;

  // Rule 4: Heart rate change
  float hn = (f.mean_rr - bl_mean) / (bl_mean + 1.0);
  if      (hn < -0.08) r += 0.10;
  else if (hn < -0.04) r += 0.05;

  // Rule 5: Regularity (rigid phenotype)
  float rg = f.regularity - bl_reg;
  if      (rg > 0.20) r += 0.10;
  else if (rg > 0.10) r += 0.05;

  // Interaction: high variance + rising = strongest signal
  if (f.norm_var > 0.8 && f.var_trend > 0.3) r += 0.10;

  return constrain(r, 0.0, 1.0);
}


// ════════════════════════════════════════════════════════════
// CALIBRATION
// ════════════════════════════════════════════════════════════
void calibrate() {
  int n = min(rr_count, MAX_RR);
  float s=0, sq=0, rs=0;
  int   rc=0;

  for (int i=0; i<n; i++) s += rr_buf[i];
  bl_mean = s / n;

  for (int i=0; i<n; i++) { float d=rr_buf[i]-bl_mean; sq+=d*d; }
  bl_var = sq / n;

  for (int i=1; i<n; i++) { float d=rr_buf[i]-rr_buf[i-1]; rs+=d*d; }
  bl_rmssd = sqrt(rs / max(n-1,1));

  for (int i=1; i<n; i++)
    if (abs(rr_buf[i]-rr_buf[i-1])/(abs(rr_buf[i-1])+0.001)<0.02) rc++;
  bl_reg = (float)rc / max(n-1,1);

  calibrated = true;

  Serial.println(F("# ╔══════════════════════════════════════╗"));
  Serial.println(F("# ║  ✅  CALIBRATION COMPLETE             ║"));
  Serial.println(F("# ╠══════════════════════════════════════╣"));
  Serial.print(F("# ║  Baseline mean RR : ")); Serial.print(bl_mean,1);
  Serial.println(F(" ms"));
  Serial.print(F("# ║  Baseline HR      : ")); Serial.print(60000.0/bl_mean,1);
  Serial.println(F(" bpm"));
  Serial.print(F("# ║  Baseline variance: ")); Serial.println(bl_var,1);
  Serial.print(F("# ║  Baseline RMSSD   : ")); Serial.print(bl_rmssd,1);
  Serial.println(F(" ms"));
  Serial.println(F("# ║  Personal model   : READY             ║"));
  Serial.println(F("# ╚══════════════════════════════════════╝"));
  Serial.println(F("# Monitoring started. Sending to ThingSpeak every 15s."));
}


// ════════════════════════════════════════════════════════════
// LED + BUZZER
// ════════════════════════════════════════════════════════════
void set_leds(int lvl) {
  digitalWrite(PIN_GREEN,  LOW);
  digitalWrite(PIN_YELLOW, LOW);
  digitalWrite(PIN_RED,    LOW);
  if      (lvl == 0) digitalWrite(PIN_GREEN,  HIGH);
  else if (lvl == 1) digitalWrite(PIN_YELLOW, HIGH);
  else if (lvl == 2) digitalWrite(PIN_YELLOW, HIGH);
  else if (lvl == 3) digitalWrite(PIN_RED,    HIGH);
}

void buzz(int lvl) {
  if      (lvl == 3) { for(int i=0;i<3;i++){tone(PIN_BUZZER,1000,100);delay(200);} }
  else if (lvl == 2) { tone(PIN_BUZZER, 800, 150); }
  else if (lvl == 1) { tone(PIN_BUZZER, 600,  80); }
}


// ════════════════════════════════════════════════════════════
// SEND DATA LINE (read by Python bridge → ThingSpeak)
// Format: DATA,<rr>,<hr>,<nvar>,<rmssd>,<risk>,<level>,<reg>,<vtrend>
// ════════════════════════════════════════════════════════════
void send_data_line() {
  Serial.print(F("DATA,"));
  Serial.print(last_rr,    1); Serial.print(',');
  Serial.print(last_hr,    1); Serial.print(',');
  Serial.print(last_nvar,  4); Serial.print(',');
  Serial.print(last_rmssd, 2); Serial.print(',');
  Serial.print(last_risk,  4); Serial.print(',');
  Serial.print(last_level);    Serial.print(',');
  Serial.print(last_reg,   4); Serial.print(',');
  Serial.println(last_vtrend, 4);
}

// Human-readable alert (prefixed # so bridge ignores it)
void print_alert(Feat f, float risk, int lvl) {
  Serial.println(F("# ──────────────────────────────────────────"));
  Serial.print(F("# Beat ")); Serial.print(rr_count);
  Serial.print(F("  HR:")); Serial.print(60000.0/f.mean_rr, 1);
  Serial.print(F("bpm  RR:")); Serial.print(f.mean_rr, 0);
  Serial.println(F("ms"));

  Serial.print(F("# "));
  if      (lvl==3) Serial.println(F("🔴 CRITICAL  — AF IMMINENT"));
  else if (lvl==2) Serial.println(F("⚠️  WARNING   — AF PREDICTED SOON"));
  else if (lvl==1) Serial.println(F("🟡 EARLY SIG — AF POSSIBLE"));
  else             Serial.println(F("✅ STABLE    — NORMAL SINUS RHYTHM"));

  Serial.print(F("# Risk=")); Serial.print(risk, 3);
  Serial.print(F("  normVar=")); Serial.print(f.norm_var, 3);
  Serial.print(F("  trend=")); Serial.println(f.var_trend, 3);
}


// ════════════════════════════════════════════════════════════
// SETUP
// ════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(9600);
  pinMode(PIN_GREEN,    OUTPUT);
  pinMode(PIN_YELLOW,   OUTPUT);
  pinMode(PIN_RED,      OUTPUT);
  pinMode(PIN_BUZZER,   OUTPUT);
  pinMode(PIN_SDN,      OUTPUT);
  pinMode(PIN_LO_PLUS,  INPUT);
  pinMode(PIN_LO_MINUS, INPUT);
  digitalWrite(PIN_SDN, HIGH);   // enable AD8232

  // Startup flash
  digitalWrite(PIN_RED,    HIGH); delay(150); digitalWrite(PIN_RED,    LOW);
  digitalWrite(PIN_YELLOW, HIGH); delay(150); digitalWrite(PIN_YELLOW, LOW);
  digitalWrite(PIN_GREEN,  HIGH); delay(150); digitalWrite(PIN_GREEN,  LOW);
  tone(PIN_BUZZER, 880, 100);

  Serial.println(F("# ╔══════════════════════════════════════════╗"));
  Serial.println(F("# ║  PERSONALISED AF MONITOR                 ║"));
  Serial.println(F("# ║  Arduino Uno + AD8232 → ThingSpeak       ║"));
  Serial.println(F("# ╠══════════════════════════════════════════╣"));
  Serial.println(F("# ║  Fields sent:                            ║"));
  Serial.println(F("# ║  1=RR(ms) 2=HR(bpm) 3=normVar 4=RMSSD   ║"));
  Serial.println(F("# ║  5=Risk   6=Level   7=Reg     8=Trend    ║"));
  Serial.println(F("# ╚══════════════════════════════════════════╝"));
  Serial.println(F("# Phase 1: CALIBRATION — collecting 60 beats"));
  Serial.println(F("# (Run af_thingspeak_bridge.py on your PC now)"));

  digitalWrite(PIN_GREEN, HIGH);
  last_send_ms = millis();
}


// ════════════════════════════════════════════════════════════
// MAIN LOOP
// ════════════════════════════════════════════════════════════
void loop() {
  unsigned long now = millis();
  float new_rr = 0;
  bool  got_beat = false;

  // ── READ NEW BEAT ───────────────────────────────────────
  if (SIM_MODE) {
    unsigned long interval = (unsigned long)(sim_next_rr() * 0.3);
    if (now - sim_last >= interval) {
      new_rr   = (float)(now - sim_last) / 0.3;
      sim_last = now;
      got_beat = true;
    }
  } else {
    // Check lead-off
    if (digitalRead(PIN_LO_PLUS) || digitalRead(PIN_LO_MINUS)) {
      static unsigned long lo_last = 0;
      if (now - lo_last > 2000) {
        Serial.println(F("# ⚠  Lead off — reattach electrodes"));
        lo_last = now;
      }
      set_leds(0);
      return;
    }
    // R-peak detection (simple threshold on A0)
    int ecg = analogRead(PIN_ECG_IN);
    int thresh = 600;  // adjust: 0–1023 scale, ~600 = R-peak for AD8232
    if (ecg > thresh && !above_thresh) {
      above_thresh = true;
      if (peak_ms > 0) {
        float rr = (float)(now - peak_ms);
        if (rr > 300 && rr < 2000) {
          new_rr   = rr;
          got_beat = true;
        }
      }
      peak_ms = now;
    } else if (ecg < thresh - 50) {
      above_thresh = false;
    }
  }

  // ── PROCESS BEAT ────────────────────────────────────────
  if (got_beat) {
    rr_buf[rr_head] = new_rr;
    rr_head = (rr_head + 1) % MAX_RR;
    rr_count++;

    if (!calibrated) {
      Serial.print('.');
      if (rr_count % 20 == 0) {
        Serial.print(F(" (")); Serial.print(rr_count);
        Serial.print('/'); Serial.print(CALIB_BEATS); Serial.println(')');
      }
      if (rr_count >= CALIB_BEATS) calibrate();
    }

    if (calibrated && rr_count % EVAL_EVERY == 0) {
      // Fill window
      float win[WINDOW_SIZE];
      int   n = min(rr_count, WINDOW_SIZE);
      for (int i = 0; i < n; i++) {
        int idx = (rr_head - n + i + MAX_RR) % MAX_RR;
        win[i]  = rr_buf[idx];
      }

      Feat  f   = extract(win, n);
      float r   = score(f);
      int   lvl = 0;
      if      (r >= 0.70) lvl = 3;
      else if (r >= 0.55) lvl = 2;
      else if (r >= 0.40) lvl = 1;

      // Cache latest values for send
      last_rr    = f.mean_rr;
      last_hr    = 60000.0 / f.mean_rr;
      last_nvar  = f.norm_var;
      last_rmssd = f.rmssd;
      last_risk  = r;
      last_level = lvl;
      last_reg   = f.regularity;
      last_vtrend= f.var_trend;

      set_leds(lvl);
      if (lvl != last_level || lvl >= 2) buzz(lvl);
      print_alert(f, r, lvl);
    }
  }

  // ── SEND TO THINGSPEAK (every 15s via Python bridge) ────
  if (calibrated && (now - last_send_ms >= SEND_EVERY_MS)) {
    last_send_ms = now;
    send_data_line();
  }
}
