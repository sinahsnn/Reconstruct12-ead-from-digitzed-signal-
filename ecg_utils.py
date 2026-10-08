import os
import wfdb
import logging
import torch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def get_hea_file_paths(directory):
    """
    Get all .hea file paths from a given directory and its subdirectories.

    Args:
        directory (str): The root directory to search in.

    Returns:
        List[str]: A list of full paths to .hea files.
    """
    hea_paths = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.hea'):
                full_path = os.path.join(dirpath, filename)
                hea_paths.append(full_path)
    return hea_paths


# Function to load a single ECG record
# def load_ecg(record_path, selected_leads=["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]):
#     """
#     Load ECG signal from a .hea record, selecting only specified leads.

#     Args:
#         record_path (str): Path to .hea file (or base path).
#         selected_leads (List[str], optional): List of lead names to extract.

#     Returns:
#         signal (np.ndarray): ECG signal of shape (samples, selected_channels)
#         record: Full WFDB record object
#     """
#     if record_path.endswith('.hea'):
#         record_path = record_path[:-4]

#     record = wfdb.rdrecord(record_path)
#     signal = record.p_signal

#     if selected_leads is not None:
#         sig_names = record.sig_name
        
#         lead_indices = [i for i, name in enumerate(sig_names) if name in selected_leads]
#         signal = signal[:, lead_indices]

#     return signal, record


def load_ecg(record_path, selected_leads=["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]):
    if record_path.endswith('.hea'):
        record_path = record_path[:-4]

    record = wfdb.rdrecord(record_path)
    signal = record.p_signal

    if selected_leads is not None:
        sig_names = list(record.sig_name)
        name_to_idx = {name: k for k, name in enumerate(sig_names)}

        # preserve requested order, and optionally fail loudly if a lead is missing
        lead_indices = [name_to_idx[name] for name in selected_leads if name in name_to_idx]

        # If you want strict behavior instead:
        # missing = [name for name in selected_leads if name not in name_to_idx]
        # if missing: raise ValueError(f"Missing leads {missing} in record {record_path}. Available: {sig_names}")
        # lead_indices = [name_to_idx[name] for name in selected_leads]

        signal = signal[:, lead_indices]

    return signal, record


# ============================================================
# Signal Evaluation Utilities
# ============================================================

import numpy as np

def compute_snr(reference, estimate, per_channel=False):
    """
    Compute SNR (Signal-to-Noise Ratio) in dB between two signals.

    Args:
        reference (np.ndarray): Original signal, shape (samples,) or (samples, channels).
        estimate (np.ndarray): Reconstructed signal.
        per_channel (bool): If True, compute SNR per channel.

    Returns:
        float or np.ndarray: SNR in decibels (dB). Scalar if 1D, or array if per_channel=True.
    """
    if per_channel and reference.ndim == 2:
        signal_power = np.mean(reference ** 2, axis=0)
        noise_power = np.mean((reference - estimate) ** 2, axis=0)
        with np.errstate(divide='ignore'):
            snr = 10 * np.log10(signal_power / noise_power)
        snr[np.isnan(snr)] = float('inf')
        return snr
    else:
        signal_power = np.mean(reference ** 2)
        noise_power = np.mean((reference - estimate) ** 2)
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)


import numpy as np
from scipy.signal import filtfilt

def lp_filter_zero_phase(x, fc):
    """
    Second-order zero-phase low-pass filter.
    """
    if fc >= 1:
        raise ValueError('fc should be smaller than 1')

    k = 1 / np.sqrt(2.0)
    alpha = (1 - k * np.cos(2 * np.pi * fc) - np.sqrt(2 * k * (1 - np.cos(2 * np.pi * fc)) - k**2 * np.sin(2 * np.pi * fc)**2)) / (1 - k)

    y = filtfilt([1 - alpha], [1, -alpha], x, axis=-1)
    return y

def preprocess_data(loadedData, fs):
    """
    Applies bandpass-like filtering to remove DC and high-freq noise.
    """
    numSamples, numChannels = loadedData.shape
    filtered_data = np.zeros((numSamples, numChannels))

    hf_thr = 0.5   # High-pass cutoff (remove DC)
    lf_thr = 80    # Low-pass cutoff (remove high-frequency noise)

    loadedData_tr = loadedData.T
    loadedData_tr_dc = lp_filter_zero_phase(loadedData_tr, hf_thr / fs)
    loadedData_tr = lp_filter_zero_phase(loadedData_tr, lf_thr / fs)
    loadedData_tr = loadedData_tr - loadedData_tr_dc

    filtered_data = loadedData_tr.T
    return filtered_data


