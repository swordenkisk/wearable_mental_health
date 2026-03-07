#!/usr/bin/env python3
"""
Demo of the wearable mental health monitor.
Simulates 24 hours of data, runs feature extraction, trains a tiny model,
and plots stress probability over time.
"""

import numpy as np
import matplotlib.pyplot as plt

from core import (
    simulate_sensor_data,
    extract_features,
    create_tiny_model,
    predict_stress,
    should_alert
)


def generate_training_labels(features):
    hr = features[:, 0]
    gsr = features[:, 3]
    return ((hr > 85) & (gsr > 5)).astype(int)


def main():
    print("Wearable Mental Health Monitor - Demo")
    print("Simulating 24 hours of sensor data at 10 Hz...")
    t, hr, gsr, acc = simulate_sensor_data(duration_hours=24, fs=10)

    print("Extracting features over 5-minute windows...")
    features = extract_features(hr, gsr, acc, fs=10, window=300)
    print(f"  -> {len(features)} windows")

    print("Generating synthetic stress labels (for training)...")
    y = generate_training_labels(features)

    print("Creating and training tiny 1D CNN...")
    model = create_tiny_model(input_dim=features.shape[1])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(features, y, epochs=5, batch_size=32, verbose=1)

    print("Running inference on entire day...")
    stress_probs = predict_stress(model, features)

    # One point per window in minutes
    window_times = np.arange(len(features)) * 5

    # Ensure we have exactly len(features) windows
    n_windows = len(features)
    window_size = 3000  # 5 minutes at 10 Hz (3000 samples)

    # Compute HR and GSR per 5‑minute window
    max_i = min(len(hr) - window_size, (n_windows - 1) * window_size)
    windows_hr = [np.mean(hr[i:i+window_size]) for i in range(0, max_i, window_size)]
    windows_gsr = [np.mean(gsr[i:i+window_size]) for i in range(0, max_i, window_size)]

    # Expand or truncate to exactly n_windows
    windows_hr = np.resize(windows_hr, (n_windows,))
    windows_gsr = np.resize(windows_gsr, (n_windows,))

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(window_times, windows_hr, label='HR (bpm, window avg)', alpha=0.7)
    plt.plot(window_times, windows_gsr, label='GSR (uS, window avg)', alpha=0.7)
    plt.legend()
    plt.ylabel('Sensor values')
    plt.title('Simulated sensor data (5-min windows)')

    plt.subplot(2, 1, 2)
    plt.plot(window_times, stress_probs, 'r-', linewidth=2, label='Stress probability')
    plt.axhline(y=0.7, color='k', linestyle='--', label='Alert threshold')
    plt.fill_between(window_times, 0.7, 1, where=(stress_probs > 0.7), color='red', alpha=0.3)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Probability')
    plt.legend()
    plt.title('Real-time stress detection')

    plt.tight_layout()
    plt.show()

    alerts = [i for i, p in enumerate(stress_probs) if p > 0.7]
    if alerts:
        print("Stress alerts triggered at times (minutes):")
        for i in alerts[:5]:
            print(f"   t = {i*5} min (p={stress_probs[i]:.2f})")
        if len(alerts) > 5:
            print(f"   ... and {len(alerts)-5} more.")
    else:
        print("No stress alerts today.")

    print("Demo complete. This invention is ready for patent protection.")


if __name__ == "__main__":
    main()
