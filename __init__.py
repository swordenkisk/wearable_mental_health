"""
Wearable Mental Health Monitor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A continuous biometric wristband using HRV, skin conductance, and sleep cycles
to detect early burnout and anxiety episodes with on‑device TinyML.

Author: swordenkisk 🇩🇿
Date:   March 2026
"""

from .core import (
    simulate_sensor_data,
    extract_features,
    create_tiny_model,
    predict_stress,
    should_alert
)

__all__ = [
    'simulate_sensor_data',
    'extract_features',
    'create_tiny_model',
    'predict_stress',
    'should_alert'
]
