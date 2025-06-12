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
    st.image("data:image/png;base64,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", use_column_width=True)
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
