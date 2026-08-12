============================================================
 EDGE DEPLOYMENT — RASPBERRY PI + AD8232
 Personalised AF Monitor
============================================================

HARDWARE NEEDED:
  - Raspberry Pi 4 (2GB RAM) ~$35
  - AD8232 ECG module         ~$8
  - MCP3008 ADC chip          ~$4
  - ECG electrode pads x3     ~$5
  - Jumper wires              ~$3
  - 16GB microSD card         ~$8
  TOTAL: ~$63

WIRING (see wiring_and_setup.md for full diagram):
  AD8232 OUTPUT → MCP3008 CH0
  MCP3008 SPI   → Raspberry Pi GPIO (SPI pins)
  AD8232 SDN    → Pi GPIO 17 (keep HIGH)
  Electrodes: RED=right chest, YELLOW=left chest, GREEN=abdomen

FOUR FILES:
  step1_install.sh     — run once to install all dependencies
  step2_ecg_reader.py  — test ECG hardware, verify R-peak detection
  step3_monitor.py     — MAIN MONITOR (calibrate + real-time predict)
  step4_autostart.sh   — make monitor start on Pi boot

HOW TO USE:
  1. Flash Raspberry Pi OS to SD card (use Raspberry Pi Imager)
  2. Boot Pi, open terminal
  3. sudo bash step1_install.sh
  4. sudo reboot
  5. Attach electrodes, connect AD8232
  6. python3 step2_ecg_reader.py           ← test hardware
  7. python3 step3_monitor.py --mode auto --patient yourname
     (6 hour calibration → then real-time monitoring starts)
  8. Optional: sudo bash step4_autostart.sh  ← auto-start on boot

SIMULATION MODE (test without hardware):
  Copy any .npz file from ltaf_project/data/rr/ to ~/af_monitor/recordings/
  python3 step3_monitor.py --mode simulate --patient rec06 --file ~/af_monitor/recordings/06_rr.npz

ALERT OUTPUTS:
  Screen: coloured text boxes with risk score
  GPIO LED (optional): Red=CRITICAL, Yellow=WARNING, Green=STABLE
  GPIO Buzzer (optional): beeps on CRITICAL
  CSV log: ~/af_monitor/logs/ (timestamp, RR, risk score, alert level)

EXPECTED PERFORMANCE:
  After 6h calibration, same patient: AUC ~0.90+
  First CRITICAL alert: ~25 minutes before AF
  Alert interval: every 30 seconds
============================================================
