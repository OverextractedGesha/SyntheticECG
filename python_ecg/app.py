import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from core import generate_spectrum, generate_rr_tachogram, solve_ecg

st.set_page_config(layout="wide", page_title="ECG Synthetic Generator (Biomodelling)")

st.title("ECG Synthetic Generator (Tugas Biomodelling)")
st.markdown("Implementasi model Dinamik McSharry dengan 4 Modifikasi: **VLF, Spektral Noise, Modulasi Slide 20, Solver ODE**")

with st.sidebar:
    st.header("1. Parameter Detak Jantung")
    hmean = st.slider("Mean Heart Rate (bpm)", 40, 120, 60)
    hstd = st.slider("HR Std Dev (bpm)", 0.1, 5.0, 1.0)
    Nrr = st.selectbox("N (Jumlah Titik RR)", [128, 256, 512, 1024], index=1)
    
    st.header("2. Modifikasi Spektrum")
    vlf_enabled = st.checkbox("Aktifkan VLF (Very Low Freq)", value=True)
    f_vlf = st.number_input("Frekuensi Pusat VLF (Hz)", value=0.02, step=0.01)
    vlf_ratio = st.slider("Bobot/Power VLF", 0.0, 1.0, 0.3)
    
    noise_level = st.slider("White Noise Level", 0.0, 0.2, 0.05, 0.01)
    
    st.header("3. Parameter Morfologi")
    mode = st.radio("Mode Modulasi Faktor", ["Bu Nada (5-point)", "McSharry (6-point)"])
    
    st.header("4. Metode Integrasi (ODE)")
    solver = st.selectbox("Solver", ["Euler", "RK2 (Heun)", "RK4", "RK8 (8-Stage)"], index=2)
    fecg = st.number_input("Frekuensi Sampling ECG (Hz)", 100, 1000, 256)

# Pembangkit Spektrum & RR Tachogram
f, S = generate_spectrum(Nrr=Nrr, hmean=hmean, hstd=hstd, 
                         vlf_enabled=vlf_enabled, f_vlf=f_vlf, vlf_ratio=vlf_ratio,
                         noise_level=noise_level)
rr = generate_rr_tachogram(S, Nrr, hmean, hstd)

st.subheader("Visualisasi Spektrum & RR Tachogram")
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(f, S, color='blue')
    ax1.set_title("Power Spectrum S(f)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Power")
    ax1.grid(True)
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(rr, color='red', marker='o', markersize=3, linestyle='-')
    ax2.set_title("RR Tachogram Time Series")
    ax2.set_xlabel("Beats (n)")
    ax2.set_ylabel("RR Interval (s)")
    ax2.grid(True)
    st.pyplot(fig2)

st.subheader(f"Simulasi Dinamika ECG 3D - Metode {solver}")
with st.spinner(f"Simulasi {solver} sedang berjalan..."):
    time_arr, U = solve_ecg(rr, fecg, solver=solver, mode=mode, hmean=hmean)

col3, col4 = st.columns(2)
with col3:
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.plot(U[:,0], U[:,1], U[:,2], color='green', linewidth=0.5)
    ax3.set_title("3D State Space Limit Cycle")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z (ECG)")
    st.pyplot(fig3)
    
with col4:
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    limit = min(len(time_arr), int(10 * fecg)) # Tampilkan 10 detik pertama
    ax4.plot(time_arr[:limit], U[:limit, 2], color='black', linewidth=1.2)
    ax4.set_title("Sinyal ECG (z) - 10 Detik Pertama")
    ax4.set_xlabel("Waktu (s)")
    ax4.set_ylabel("Amplitudo (mV)")
    ax4.grid(True)
    st.pyplot(fig4)
