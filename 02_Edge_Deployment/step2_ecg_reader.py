"""
STEP 2 — ECG READER + R-PEAK DETECTOR
=======================================
Reads raw ECG from AD8232 via MCP3008 SPI ADC.
Detects R-peaks (heartbeats) in real time.
Outputs RR intervals in milliseconds.

Run: python3 step2_ecg_reader.py
     (test this BEFORE running the monitor — verify you see clean beats)
"""

import time
import numpy as np
import spidev
import RPi.GPIO as GPIO
from collections import deque

# ── GPIO SETUP ───────────────────────────────────────────────────────────────
GPIO.setmode(GPIO.BCM)
SDN_PIN = 17   # AD8232 shutdown — pull HIGH to enable
LO_PLUS = 27   # lead-off detection +
LO_MINUS= 22   # lead-off detection -

GPIO.setup(SDN_PIN,  GPIO.OUT)
GPIO.setup(LO_PLUS,  GPIO.IN)
GPIO.setup(LO_MINUS, GPIO.IN)
GPIO.output(SDN_PIN, GPIO.HIGH)   # enable AD8232

# ── SPI SETUP (MCP3008) ───────────────────────────────────────────────────────
spi = spidev.SpiDev()
spi.open(0, 0)          # bus 0, device 0 (CE0)
spi.max_speed_hz = 1350000

SAMPLE_RATE = 500        # Hz — how fast we read the ADC
CHANNEL     = 0          # MCP3008 channel 0

def read_adc(channel=0):
    """Read one sample from MCP3008."""
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data   # 0 to 1023

# ── R-PEAK DETECTOR (Pan-Tompkins simplified) ─────────────────────────────────
class RPeakDetector:
    """
    Real-time R-peak detection from raw ECG.
    Uses derivative + threshold method (simplified Pan-Tompkins).
    """
    def __init__(self, fs=500):
        self.fs          = fs
        self.buf         = deque(maxlen=int(fs * 0.2))   # 200ms buffer
        self.deriv_buf   = deque(maxlen=5)
        self.threshold   = 100
        self.last_peak   = 0        # sample index of last R-peak
        self.sample_idx  = 0
        self.refractory  = int(fs * 0.25)  # 250ms refractory period
        self.signal_buf  = deque(maxlen=int(fs * 2))     # 2s for threshold adapt

    def process(self, sample):
        """
        Feed one ECG sample. Returns RR interval in ms if R-peak detected,
        else returns None.
        """
        self.sample_idx += 1
        self.buf.append(sample)
        self.signal_buf.append(sample)

        # Running baseline (slow adaptive threshold)
        if len(self.signal_buf) > self.fs:
            sig = np.array(self.signal_buf)
            self.threshold = np.mean(sig) + 0.6 * np.std(sig)

        # Derivative
        if len(self.buf) >= 5:
            d = (self.buf[-1] - self.buf[-5]) * self.fs / 4
            self.deriv_buf.append(d)

        # Peak detection: sample above threshold, derivative crossing zero
        if (len(self.deriv_buf) >= 2 and
                sample > self.threshold and
                self.deriv_buf[-2] > 0 and self.deriv_buf[-1] <= 0 and
                (self.sample_idx - self.last_peak) > self.refractory):

            rr_samples = self.sample_idx - self.last_peak
            rr_ms      = rr_samples / self.fs * 1000.0

            # Sanity check: RR must be 300–2000ms (20–200 bpm)
            if 300 < rr_ms < 2000:
                self.last_peak = self.sample_idx
                return rr_ms

        return None


def check_leads():
    """Returns True if electrodes are connected."""
    lo_plus  = GPIO.input(LO_PLUS)
    lo_minus = GPIO.input(LO_MINUS)
    if lo_plus or lo_minus:
        return False   # lead off
    return True        # leads connected


def run_ecg_test(duration_sec=30):
    """
    Test mode — run for N seconds, print detected beats.
    Use this to verify your hardware is working.
    """
    detector    = RPeakDetector(fs=SAMPLE_RATE)
    rr_list     = []
    sample_dt   = 1.0 / SAMPLE_RATE
    start_time  = time.time()

    print(f"Reading ECG for {duration_sec} seconds...")
    print("  Make sure electrodes are attached.")
    print("  You should see RR values around 600–1000ms (60–100 bpm)\n")

    try:
        while time.time() - start_time < duration_sec:
            t0 = time.time()

            if not check_leads():
                print("  ⚠  Lead off — check electrode connections")
                time.sleep(0.5)
                continue

            sample = read_adc(CHANNEL)
            rr     = detector.process(sample)

            if rr is not None:
                rr_list.append(rr)
                hr = 60000 / rr
                print(f"  Beat {len(rr_list):4d}  RR = {rr:6.1f} ms  HR = {hr:5.1f} bpm")

            # maintain sample rate
            elapsed = time.time() - t0
            sleep   = sample_dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        spi.close()
        GPIO.cleanup()

    if len(rr_list) >= 10:
        print(f"\n✅  SUCCESS — detected {len(rr_list)} beats")
        print(f"   Mean RR = {np.mean(rr_list):.1f} ms  ({60000/np.mean(rr_list):.1f} bpm)")
        print(f"   SDNN    = {np.std(rr_list):.1f} ms")
        print(f"\nHardware is working. Proceed to step3_monitor.py")
    else:
        print(f"\n❌  Only {len(rr_list)} beats detected in {duration_sec}s")
        print("   Check: electrode placement, SPI wiring, AD8232 power")

    return rr_list


if __name__ == '__main__':
    rrs = run_ecg_test(duration_sec=60)
