#!/bin/bash
# ============================================================
# STEP 1 — Run this on your Raspberry Pi after fresh OS install
# sudo bash step1_install.sh
# ============================================================

echo "=== Updating system ==="
sudo apt-get update && sudo apt-get upgrade -y

echo "=== Installing system dependencies ==="
sudo apt-get install -y python3-pip python3-dev git libatlas-base-dev

echo "=== Enabling SPI (for MCP3008 ADC) ==="
sudo raspi-config nonint do_spi 0
echo "dtparam=spi=on" | sudo tee -a /boot/config.txt

echo "=== Installing Python packages ==="
pip3 install numpy scipy scikit-learn joblib spidev RPi.GPIO

echo "=== Installing antropy for entropy features ==="
pip3 install antropy

echo "=== Verifying SPI ==="
ls /dev/spidev*

echo ""
echo "========================================"
echo "Installation complete. Reboot now:"
echo "  sudo reboot"
echo "========================================"
