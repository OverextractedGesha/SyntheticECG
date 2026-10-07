# SyntheticECG - ECG Synthetic Generator (Biomodelling)

Aplikasi simulasi dan pembangkit sinyal elektrokardiogram (ECG) sintetis interaktif berbasis **Streamlit**, menggunakan model dinamik McSharry dengan 4 modifikasi khusus.

---

## Fitur & Modifikasi

1. **Modifikasi 1 - Very Low Frequency (VLF)**:
   - Penambahan komponen frekuensi sangat rendah (VLF) pada Power Spectral Density (PSD) di samping komponen LF (Mayer wave) dan HF (Respiratory Sinus Arrhythmia / RSA).
2. **Modifikasi 2 - Spektral White Noise**:
   - Penambahan derau putih (*white noise*) pada spektrum untuk mensimulasikan karakteristik spektrum biologis yang lebih realistis.
3. **Modifikasi 3 - Modulasi Morfologi Gelombang (Slide 20)**:
   - Dukungan pemilihan metode modulasi faktor morfologi:
     - **Bu Nada (5-point)**: Titik P, Q, R, S, T standar modulasi.
     - **McSharry (6-point)**: Titik P, Q, R, S, T-, T+ untuk pemodelan gelombang T asimetris.
4. **Modifikasi 4 - Solver Numerik Persamaan Diferensial Biasa (ODE)**:
   - Komparasi integrasi numerik untuk sistem dinamik 3D limit-cycle:
     - Euler Method (Orde 1)
     - RK2 / Heun's Method (Orde 2)
     - RK4 (Runge-Kutta Orde 4 standar)
     - RK8 (8-Stage Explicit Runge-Kutta)

---

## Struktur Folder

```text
Project Bu Nada/
├── .gitignore
├── README.md
└── python_ecg/
    ├── app.py              # Antarmuka web interaktif Streamlit
    ├── core.py             # Algoritma spektrum, IDFT manual, ODE solver
    ├── requirements.txt    # Daftar dependensi Python
    └── run.ps1             # Skrip PowerShell untuk menjalankan aplikasi
```

---

## Cara Menjalankan

### 1. Prasyarat
Pastikan Python 3.8+ telah terpasang pada sistem Anda.

### 2. Instalasi Dependensi
Buka terminal / PowerShell dan arahkan ke direktori `python_ecg`:
```bash
cd python_ecg
pip install -r requirements.txt
```

### 3. Menjalankan Dashboard
Jalankan aplikasi Streamlit dengan perintah:
```bash
streamlit run app.py
```
atau menggunakan skrip PowerShell yang tersedia:
```powershell
.\run.ps1
```
Aplikasi akan terbuka otomatis di browser Anda pada alamat `http://localhost:8501`.
