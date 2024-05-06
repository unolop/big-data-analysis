from scipy import signal
import numpy as np 

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_signal = signal.lfilter(b, a, signal)
    return filtered_signal

def normalize_ecg(ecg_signal):
    mean = np.mean(ecg_signal)
    std = np.std(ecg_signal)
    normalized_signal = (ecg_signal - mean) / std
    return normalized_signal 