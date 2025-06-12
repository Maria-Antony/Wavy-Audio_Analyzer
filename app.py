# import os
# import streamlit as st
# import librosa
# import numpy as np
# import tempfile
# from analysis import load_audio, extract_features
# from visualizations import *
# from llm_initializer import get_insights

# from dotenv import load_dotenv
# load_dotenv()


# st.set_page_config(page_title="SignalScope AI", layout="wide")
# st.title("🎧 WaveLine – Audio Signal Analysis Tool backed with LLM Insights")

# uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

# if uploaded_file:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
#         tmpfile.write(uploaded_file.read())
#         tmp_path = tmpfile.name

#     y, sr = load_audio(tmp_path)
#     features = extract_features(y, sr)
#     stats = features["stats_summary"]

#     st.session_state["signal_stats"] = stats
#     st.sidebar.subheader("📊 Signal Stats")
#     for k, v in stats.items():
#         st.sidebar.write(f"**{k}**: {v:.4f}")

#     st.plotly_chart(plot_waveform(y, sr), use_container_width=True)
#     st.plotly_chart(plot_mel_spectrogram(features["mel_spectrogram"], sr), use_container_width=True)
#     st.plotly_chart(plot_zcr_rms(features["zcr"], features["rms_energy"]), use_container_width=True)
#     st.plotly_chart(plot_mfcc(features["mfcc"]), use_container_width=True)
#     st.plotly_chart(plot_chromagram(features["chroma"]), use_container_width=True)
#     st.plotly_chart(plot_tempogram(features["tempogram"]), use_container_width=True)
#     st.plotly_chart(plot_power_spectrum(features["psd_freqs"], features["psd"]), use_container_width=True)
#     st.plotly_chart(plot_wavelet(features["wavelet"]), use_container_width=True)

#     st.subheader("🧠 Mistral Insights")
#     api_key = os.getenv("MISTRAL_API_KEY")
#     if api_key:
#         try:
#             response = get_insights(stats, api_key)
#             st.success(response)
#         except Exception as e:
#             st.error(f"Error: {e}")
#     else:
#         st.warning("No MISTRAL_API_KEY found. Add it to a .env file.")

#     exec(open("signal_chatbot.py").read())

import os
import streamlit as st
import librosa
import numpy as np
import tempfile
from analysis import load_audio, extract_features
from visualizations import *
from llm_initializer import get_insights
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="🎧 WaveLine", layout="wide", page_icon="🎧")

with st.sidebar:
    st.title("🎛️ WaveLine AI")
    st.markdown("An interactive, LLM-backed signal explorer.")
    st.image("https://media.giphy.com/media/Y4z9olnoVl5QI/giphy.gif", use_column_width=True)
    st.markdown("Upload an audio file and visualize all its components. Get real-time signal interpretation using Mistral LLM.")

st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)


st.title("🎧 WaveLine – AI Signal Analyzer")

uploaded_file = st.file_uploader("📤 Upload an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file:
    with st.spinner("🔍 Analyzing your audio..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmpfile.write(uploaded_file.read())
            tmp_path = tmpfile.name

        y, sr = load_audio(tmp_path)
        features = extract_features(y, sr)
        stats = features["stats_summary"]

        st.session_state["signal_stats"] = stats

    st.sidebar.subheader("📊 Signal Statistics")
    for k, v in stats.items():
        st.sidebar.write(f"**{k}**: {v:.4f}")

    st.success("✅ Audio analyzed successfully!")

    st.subheader("🎼 Waveform")
    st.plotly_chart(plot_waveform(y, sr), use_container_width=True)

    st.subheader("🌈 Mel Spectrogram")
    st.plotly_chart(plot_mel_spectrogram(features["mel_spectrogram"], sr), use_container_width=True)

    st.subheader("📈 ZCR and RMS Energy")
    st.plotly_chart(plot_zcr_rms(features["zcr"], features["rms_energy"]), use_container_width=True)

    st.subheader("🔤 MFCC Heatmap")
    st.plotly_chart(plot_mfcc(features["mfcc"]), use_container_width=True)

    st.subheader("🎹 Chromagram")
    st.plotly_chart(plot_chromagram(features["chroma"]), use_container_width=True)

    st.subheader("⏱️ Tempogram")
    st.plotly_chart(plot_tempogram(features["tempogram"]), use_container_width=True)

    st.subheader("🔊 Power Spectrum")
    st.plotly_chart(plot_power_spectrum(features["psd_freqs"], features["psd"]), use_container_width=True)

    st.subheader("🧩 Wavelet Scalogram")
    st.plotly_chart(plot_wavelet(features["wavelet"]), use_container_width=True)

    st.subheader("🧠 Mistral Insights")
    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key:
        with st.spinner("🤖 Generating smart insights..."):
            try:
                response = get_insights(stats, api_key)
                st.success(response)
            except Exception as e:
                st.error(f"LLM Error: {e}")
    else:
        st.warning("No MISTRAL_API_KEY found. Add it to a .env file.")

    exec(open("signal_chatbot.py").read())
