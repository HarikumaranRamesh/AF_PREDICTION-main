"""
AF MONITOR — ThingSpeak Bridge
================================
Runs on your PC. Reads data from Arduino over USB Serial,
POSTs it to ThingSpeak every 15 seconds.

INSTALL:  pip install pyserial requests
RUN:      python af_thingspeak_bridge.py

BEFORE RUNNING:
  1. Set YOUR_API_KEY below
  2. Set YOUR_COM_PORT below (e.g. COM3 on Windows, /dev/ttyACM0 on Linux)
  3. Upload af_thingspeak.ino to Arduino first
  4. Run this script — keep it running while Arduino is connected
"""

import serial
import requests
import time
import sys
from datetime import datetime

# ── CONFIGURATION — FILL THESE IN ────────────────────────────
API_KEY   = "YOUR_WRITE_API_KEY"   # ← paste your ThingSpeak Write API Key
COM_PORT  = "COM3"                  # ← your Arduino port
#            Windows: COM3, COM4 etc  (check Device Manager)
#            Mac:     /dev/cu.usbmodem14101  (check ls /dev/cu.*)
#            Linux:   /dev/ttyACM0 or /dev/ttyUSB0
BAUD_RATE = 9600

# ── THINGSPEAK ────────────────────────────────────────────────
TS_URL    = "https://api.thingspeak.com/update"

# ThingSpeak field mapping:
# field1 = RR interval (ms)
# field2 = Heart rate (bpm)
# field3 = Normalised variance
# field4 = RMSSD (ms)
# field5 = Risk score (0.0–1.0)
# field6 = Alert level (0/1/2/3)
# field7 = Regularity index
# field8 = Variance trend

ALERT_NAMES = {
    "0": "✅  STABLE",
    "1": "🟡  EARLY SIGNAL",
    "2": "⚠️   WARNING",
    "3": "🔴  CRITICAL",
}

def post_to_thingspeak(fields: dict) -> bool:
    """Send one update to ThingSpeak. Returns True if successful."""
    payload = {"api_key": API_KEY}
    payload.update(fields)
    try:
        r = requests.get(TS_URL, params=payload, timeout=10)
        if r.status_code == 200 and r.text.strip() != "0":
            return True
        else:
            print(f"  ThingSpeak rejected: status={r.status_code} body={r.text.strip()}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"  Network error: {e}")
        return False

def parse_data_line(line: str) -> dict | None:
    """
    Parse a DATA line from Arduino.
    Format: DATA,<rr>,<hr>,<nvar>,<rmssd>,<risk>,<level>,<reg>,<vtrend>
    Returns dict of ThingSpeak fields, or None if invalid.
    """
    line = line.strip()
    if not line.startswith("DATA,"):
        return None
    parts = line[5:].split(',')
    if len(parts) < 8:
        return None
    try:
        return {
            "field1": float(parts[0]),   # RR interval
            "field2": float(parts[1]),   # Heart rate
            "field3": float(parts[2]),   # Normalised variance
            "field4": float(parts[3]),   # RMSSD
            "field5": float(parts[4]),   # Risk score
            "field6": int(parts[5]),     # Alert level
            "field7": float(parts[6]),   # Regularity index
            "field8": float(parts[7]),   # Variance trend
        }
    except ValueError:
        return None

def main():
    print("=" * 55)
    print("  AF MONITOR — ThingSpeak Bridge")
    print("=" * 55)

    # ── Validate config ───────────────────────────────────────
    if API_KEY == "YOUR_WRITE_API_KEY":
        print("\n  ❌  ERROR: Paste your ThingSpeak Write API Key")
        print("     Edit af_thingspeak_bridge.py line 22")
        sys.exit(1)

    # ── Connect to Arduino ────────────────────────────────────
    print(f"\n  Connecting to Arduino on {COM_PORT} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)   # wait for Arduino reset after serial connect
        ser.reset_input_buffer()
        print(f"  ✅  Connected to {COM_PORT}")
    except serial.SerialException as e:
        print(f"\n  ❌  Cannot open {COM_PORT}: {e}")
        print("\n  How to find your Arduino port:")
        print("    Windows: Device Manager → Ports (COM & LPT)")
        print("    Mac:     ls /dev/cu.*")
        print("    Linux:   ls /dev/ttyACM* or ls /dev/ttyUSB*")
        sys.exit(1)

    print(f"\n  ThingSpeak API Key: {API_KEY[:6]}...")
    print(f"  Sending every 15 seconds (ThingSpeak free limit)")
    print(f"  Dashboard: https://thingspeak.com/channels/YOUR_CHANNEL_ID")
    print(f"\n  Waiting for Arduino calibration to complete...")
    print(f"  (Lines starting with # are Arduino log messages)\n")
    print("-" * 55)

    # ── Main loop ─────────────────────────────────────────────
    posts_sent    = 0
    posts_failed  = 0
    last_data     = None

    try:
        while True:
            # Read one line from Arduino
            try:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode('utf-8', errors='ignore').strip()
            except serial.SerialException as e:
                print(f"\n  ❌  Serial error: {e}")
                break

            if not line:
                continue

            # Print Arduino log messages (# prefix)
            if line.startswith('#'):
                print(f"  Arduino | {line[2:]}")
                continue

            # Parse DATA line
            if line.startswith('DATA,'):
                fields = parse_data_line(line)
                if fields:
                    last_data  = fields
                    ts_now     = datetime.now().strftime("%H:%M:%S")
                    alert_name = ALERT_NAMES.get(str(int(fields["field6"])), "?")

                    print(f"\n  [{ts_now}]  Sending to ThingSpeak...")
                    print(f"    RR={fields['field1']:.0f}ms  "
                          f"HR={fields['field2']:.1f}bpm  "
                          f"Risk={fields['field5']:.3f}  "
                          f"{alert_name}")

                    ok = post_to_thingspeak(fields)
                    if ok:
                        posts_sent += 1
                        print(f"    ✅  Sent  (total: {posts_sent})")
                    else:
                        posts_failed += 1
                        print(f"    ❌  Failed (total failed: {posts_failed})")
                else:
                    print(f"  ⚠  Bad DATA line: {line}")

    except KeyboardInterrupt:
        print(f"\n\n  Stopped by user.")
        print(f"  Posts sent:   {posts_sent}")
        print(f"  Posts failed: {posts_failed}")
    finally:
        ser.close()
        print(f"  Serial port closed.")

if __name__ == '__main__':
    main()
