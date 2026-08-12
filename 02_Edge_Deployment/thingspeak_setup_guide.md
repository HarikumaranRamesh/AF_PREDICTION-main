# ThingSpeak Channel Setup — Step by Step

## Step 1 — Create the Channel

1. Go to thingspeak.com → Sign in
2. Click "Channels" → "New Channel"
3. Fill in:
   - Name: "Personalised AF Monitor"
   - Description: "Real-time AF prediction from Arduino + AD8232"

## Step 2 — Add These 8 Fields (exact names matter for dashboard)

| Field # | Name               | Unit  |
|---------|--------------------|-------|
| Field 1 | RR Interval        | ms    |
| Field 2 | Heart Rate         | bpm   |
| Field 3 | Normalised Variance| —     |
| Field 4 | RMSSD              | ms    |
| Field 5 | Risk Score         | 0–1   |
| Field 6 | Alert Level        | 0–3   |
| Field 7 | Regularity Index   | —     |
| Field 8 | Variance Trend     | —     |

Click "Save Channel"

## Step 3 — Get Your Write API Key

1. Click "API Keys" tab on your channel page
2. Copy the "Write API Key" (looks like: ABCDEF1234567890)
3. Paste it into af_thingspeak_bridge.py line 22

## Step 4 — Set Up Dashboard Visualisations

Go to "Private View" tab → Add Visualisations:

### Gauge — Risk Score (most important)
- Add Widget → "Gauge"
- Field: Field 5 (Risk Score)
- Min: 0,  Max: 1
- Colour: Green (0–0.4), Yellow (0.4–0.7), Red (0.7–1.0)

### Line Chart — RR Interval
- Add Widget → "Line Chart"
- Field: Field 1 (RR Interval)
- Title: "RR Interval (ms)"
- Y-axis: 300 to 1200

### Line Chart — Risk Score over Time
- Add Widget → "Line Chart"
- Field: Field 5 (Risk Score)
- Title: "AF Risk Score"
- Y-axis: 0 to 1
- Add horizontal line at 0.70 (critical threshold)

### Numeric Display — Alert Level
- Add Widget → "Numeric Display"
- Field: Field 6 (Alert Level)
- Title: "Alert (0=Stable 3=Critical)"

### Line Chart — Heart Rate
- Add Widget → "Line Chart"
- Field: Field 2 (Heart Rate)
- Title: "Heart Rate (bpm)"

## Step 5 — Share Your Dashboard (for presentation)

1. Click "Sharing" tab
2. Select "Share channel view with everyone"
3. Copy the public URL
4. Anyone can see your live monitor at this URL — no login needed

## Alert Level Meanings (Field 6)
- 0 = ✅  STABLE        — normal sinus rhythm
- 1 = 🟡  EARLY SIGNAL  — mild deviation from baseline
- 2 = ⚠️   WARNING       — AF predicted soon
- 3 = 🔴  CRITICAL      — AF imminent
