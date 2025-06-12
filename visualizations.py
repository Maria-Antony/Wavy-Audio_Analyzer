import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import librosa

def plot_waveform(y, sr):
    t = np.linspace(0, len(y) / sr, len(y))
    return go.Figure([go.Scatter(x=t, y=y)]).update_layout(title="Waveform", xaxis_title="Time", yaxis_title="Amplitude")

def plot_mel_spectrogram(S, sr):
    S_dB = librosa.power_to_db(S, ref=np.max)
    return px.imshow(S_dB, origin="lower", title="Mel Spectrogram (dB)")

def plot_zcr_rms(zcr, rms):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=zcr, name="ZCR"))
    fig.add_trace(go.Scatter(y=rms, name="RMS"))
    return fig.update_layout(title="ZCR & RMS", xaxis_title="Frame")

def plot_mfcc(mfcc):
    return px.imshow(mfcc, origin="lower", title="MFCC Heatmap")

def plot_chromagram(chroma):
    return px.imshow(chroma, origin="lower", title="Chromagram")

def plot_tempogram(tempogram):
    return px.imshow(tempogram, origin="lower", title="Tempogram")

def plot_power_spectrum(freqs, psd):
    return go.Figure([go.Scatter(x=freqs, y=psd)]).update_layout(title="Power Spectrum", xaxis_title="Hz", yaxis_title="Power")

def plot_wavelet(coeff_array):
    import plotly.express as px
    from numpy import abs
    fig = px.imshow(np.abs(coeff_array), origin='lower', title="Wavelet Transform (DWT)")
    return fig

