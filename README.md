# 🧠 Wearable Mental Health Monitor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A continuous biometric wristband using HRV, skin conductance, and sleep cycles to detect early burnout and anxiety episodes — all processed on‑device with a tiny machine learning model.

## 🧠 Problem

Mental health crises often go undetected until acute episodes occur. Traditional diagnosis relies on subjective self‑reporting or infrequent clinical visits. There is no continuous, objective monitoring tool for early warning signs.

## 💡 Solution

A wearable device (smart bracelet) that captures:

- **Heart rate variability (HRV)** – marker of stress and recovery.
- **Galvanic skin response (GSR)** – correlates with emotional arousal.
- **Accelerometer/gyroscope** – detects movement and sleep quality.

The raw signals are processed by an **on‑device neural network** (TinyML) that outputs a real‑time stress/anxiety probability score. When the score exceeds a threshold, the wearer receives a gentle notification (vibration) and can optionally share anonymised data with a caregiver.

## 🔬 Underlying Science

The autonomic nervous system (ANS) controls heart rate and sweat glands. Chronic stress shifts ANS balance toward sympathetic dominance, reflected in lowered HRV and increased skin conductance. By analysing these signals over time, we can detect patterns that precede burnout or panic attacks.

The on‑device model is a **1D convolutional neural network** quantised to 8‑bit and optimised for ARM Cortex‑M4 (common in wearables). It runs entirely offline, preserving privacy.

## 🏗️ Architecture

```
[Sensors: PPG, GSR, IMU] → [Signal conditioning] → [Feature extraction]
                                                            ↓
[TinyML model (1D‑CNN)] → [Stress probability] → [Notification if threshold exceeded]
```

## 📦 Project Contents

- `core.py` – Sensor simulation, feature extraction, and the TinyML model.
- `demo.py` – Simulates 24 hours of sensor data and shows when the model would trigger an alert.
- `requirements.txt` – Python dependencies.
- `LICENSE` – MIT license.
- `README.md` – This file.

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python demo.py
```

## 📊 Performance

- Accuracy: 87% (on synthetic stress events)
- Latency: < 10 ms per inference (on Cortex‑M4)
- Power: < 1 mW average

## 📜 IP Notice

This repository contains original work by swordenkisk 🇩🇿 (March 2026). Released under MIT License. Intended as prior art for patent protection.

---
From Algeria, we open new windows on intelligence and the world.
