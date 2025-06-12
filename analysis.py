import librosa
import numpy as np
import scipy.stats as stats
from scipy.signal import welch
import pywt

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def extract_features(y, sr):
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    rms_energy = librosa.feature.rms(y=y)[0]
    f0 = librosa.yin(y, fmin=50, fmax=500)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr)
    freqs, psd = welch(y, sr)

    stats_summary = {
        "Duration (s)": round(librosa.get_duration(y=y, sr=sr), 2),
        "Mean Amplitude": float(np.mean(y)),
        "ZCR Mean": float(np.mean(zcr)),
        "RMS Energy Mean": float(np.mean(rms_energy)),
        "Entropy": float(-np.sum(np.square(y) * np.log(np.square(y) + 1e-8)))
    }

    return {
        "zcr": zcr,
        "rms_energy": rms_energy,
        "f0": f0,
        "mfcc": mfcc,
        "mel_spectrogram": S,
        "chroma": chroma,
        "tempogram": tempogram,
        "psd_freqs": freqs,
        "psd": psd,
        "stats_summary": stats_summary
    }
