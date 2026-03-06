"""
Core implementation of the wearable mental health monitor.
Simulates sensor data, extracts features, and runs a TinyML model.
"""

import numpy as np
import scipy.signal as signal
import tensorflow as tf
from tensorflow.keras import layers, models

# ------------------------------------------------------------------------------
# Sensor simulation
# ------------------------------------------------------------------------------

def simulate_sensor_data(duration_hours=24, fs=10):
    """
    Simulate PPG (for HRV), GSR, and accelerometer data for a 24‑hour period.
    Returns:
        t: time vector (seconds)
        hr: heart rate (bpm)
        gsr: galvanic skin response (microsiemens)
        acc: 3‑axis acceleration (g)
    """
    t = np.arange(0, duration_hours * 3600, 1/fs)
    n = len(t)

    # Heart rate: normal variation (60‑80 bpm) with occasional stress spikes
    hr = 70 + 5 * np.sin(2 * np.pi * t / 86400)
    hr += 10 * np.exp(-((t - 6*3600) / 3600)**2)
    hr += 15 * np.exp(-((t - 18*3600) / 3600)**2)
    hr += np.random.randn(n) * 2

    # GSR: baseline 2‑4 µS, increases during stress
    gsr = 3 + 1.5 * np.sin(2 * np.pi * t / 86400)
    gsr += 2 * np.exp(-((t - 6*3600) / 3600)**2)
    gsr += 2.5 * np.exp(-((t - 18*3600) / 3600)**2)
    gsr = np.clip(gsr + np.random.randn(n)*0.2, 0.5, 10)

    # Accelerometer: simulate activity (walking, resting)
    acc_mag = 1.0 + 0.2 * np.sin(2 * np.pi * t / 3600)
    acc_mag += 0.5 * (np.random.rand(n) < 0.05)
    acc = np.column_stack([acc_mag, acc_mag*0.3, acc_mag*0.1]) * 0.1

    return t, hr, gsr, acc


def extract_features(hr, gsr, acc, fs, window=300):
    """
    Extract features over a moving window (default 5 minutes).
    Returns: feature matrix (samples x features)
    """
    step = fs * 60
    win_len = fs * window
    n = len(hr)
    features = []

    for start in range(0, n - win_len, step):
        end = start + win_len
        hr_win = hr[start:end]
        gsr_win = gsr[start:end]
        acc_win = acc[start:end]

        hr_mean = np.mean(hr_win)
        hr_std = np.std(hr_win)
        rmssd = np.sqrt(np.mean(np.diff(hr_win)**2))

        gsr_mean = np.mean(gsr_win)
        gsr_std = np.std(gsr_win)

        acc_mag = np.linalg.norm(acc_win, axis=1)
        acc_mean = np.mean(acc_mag)
        acc_std = np.std(acc_mag)

        features.append([hr_mean, hr_std, rmssd, gsr_mean, gsr_std, acc_mean, acc_std])

    return np.array(features)


# ------------------------------------------------------------------------------
# TinyML model (1D CNN)
# ------------------------------------------------------------------------------

def create_tiny_model(input_dim=7):
    """
    Build a small 1D CNN suitable for on‑device inference.
    """
    model = models.Sequential([
        layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
        layers.Conv1D(8, 3, activation='relu', padding='same'),
        layers.AveragePooling1D(2),
        layers.Conv1D(16, 3, activation='relu', padding='same'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def train_model(X, y):
    model = create_tiny_model(X.shape[1])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2, verbose=0)
    return model


def predict_stress(model, features):
    return model.predict(features, verbose=0).flatten()


def should_alert(prob, threshold=0.7):
    return prob > threshold
