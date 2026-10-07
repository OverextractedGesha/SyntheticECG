import numpy as np

def generate_spectrum(Nrr=256, hmean=60, hstd=1, lf_hf_ratio=0.5,
                      vlf_enabled=True, f_vlf=0.02, c_vlf=0.01, vlf_ratio=0.3,
                      f_lf=0.1, c_lf=0.01, f_hf=0.25, c_hf=0.01, noise_level=0.0):
    """
    Modifikasi 1 & 2: Pembangkit spektrum VLF, LF, HF dan tambahan white noise.
    """
    f = np.arange(Nrr) * (1.0 / Nrr) # fs = 1 Hz
    magSf = 1.7 / Nrr
    
    # 1. LF and HF (Mayer & RSA)
    sf_lf = lf_hf_ratio * magSf * np.exp(-((f - f_lf)**2) / (2 * c_lf**2)) / np.sqrt(2 * np.pi * c_lf**2)
    sf_hf = magSf * np.exp(-((f - f_hf)**2) / (2 * c_hf**2)) / np.sqrt(2 * np.pi * c_hf**2)
    S = sf_lf + sf_hf
    
    # 2. Tambahan VLF (Very Low Frequency)
    if vlf_enabled:
        sf_vlf = vlf_ratio * magSf * np.exp(-((f - f_vlf)**2) / (2 * c_vlf**2)) / np.sqrt(2 * np.pi * c_vlf**2)
        S += sf_vlf
        
    # 3. Tambahan White Noise (Spektra lebih realistis)
    if noise_level > 0:
        noise = np.random.uniform(0, noise_level * np.max(S), size=Nrr)
        S += noise
        
    return f, S

def generate_rr_tachogram(S, Nrr, hmean, hstd):
    """
    Menghasilkan deret waktu RR interval (RR tachogram).
    """
    Sw = np.zeros(Nrr)
    half_N = Nrr // 2
    for i in range(half_N):
        Sw[i] = np.sqrt(S[i])
    for i in range(half_N, Nrr):
        Sw[i] = np.sqrt(S[Nrr - i])
        
    phases = 2 * np.pi * np.random.rand(Nrr)
    phases[0] = 0.0
    if Nrr % 2 == 0:
        phases[half_N] = 0.0
    for i in range(1, half_N):
        phases[Nrr - i] = -phases[i]
        
    # IDFT Manual (Sesuai dengan Algoritma Bu Nada Step 4)
    # Tanpa menggunakan library np.fft.ifft
    idft_result = np.zeros(Nrr)
    Reall = Sw * np.cos(phases)
    Imag = Sw * np.sin(phases)
    
    for n in range(Nrr):
        re_sum = 0.0
        im_sum = 0.0
        for k in range(Nrr):
            angle = 2.0 * np.pi * k * n / Nrr
            re_sum += Reall[k] * np.cos(angle)
            im_sum += Imag[k] * np.sin(angle)
        idft_result[n] = (re_sum + im_sum) / Nrr
        
    if np.std(idft_result) > 0:
        idft_result = (idft_result - np.mean(idft_result)) / np.std(idft_result)
    
    rrmean = 60.0 / hmean
    rr_std = hstd / 60.0 
    rr = rrmean + idft_result * rr_std
    return rr

def get_morphology_parameters(hmean, mode):
    """
    Modifikasi 3: Pemilihan Modulation Factor berdasarkan Slide 20
    """
    alpha = np.sqrt(hmean / 60.0)
    
    if mode == "Bu Nada (5-point)":
        hrfact = np.sqrt(hmean / 60.0)
        hrfact2 = np.sqrt(hrfact)
        
        ti = np.array([-0.2, -0.05, 0.0, 0.05, 0.3])
        ai = np.array([1.2, -5.0, 30.0, -7.5, 0.75])
        bi = np.array([0.25, 0.1, 0.1, 0.1, 0.4])
        theta_i = np.array([-60, -15, 0, 15, 90]) * np.pi / 180.0
        
        ti[0] *= hrfact2
        ti[1] *= hrfact
        ti[3] *= hrfact
        ti[4] *= hrfact2
        bi *= hrfact2
        return ti, ai, bi, theta_i
        
    else: # "McSharry (6-point)" - Asymmetric T-Wave
        ti = np.array([
            -0.2 * np.sqrt(alpha),
            -0.05 * alpha,
            0.0,
            0.05 * alpha,
            0.277 * np.sqrt(alpha),
            0.286 * np.sqrt(alpha)
        ])
        theta_i = np.array([
            -(np.pi * np.sqrt(alpha)) / 3.0,
            -(np.pi * alpha) / 12.0,
            0.0,
            (np.pi * alpha) / 12.0,
            (5 * np.pi * np.sqrt(alpha) / 9.0) - (np.pi * np.sqrt(alpha) / 60.0),
            (5 * np.pi * np.sqrt(alpha)) / 9.0
        ])
        ai = np.array([
            0.8,
            -5.0,
            30.0,
            -7.5,
            0.5 * (alpha ** 2.5),
            0.75 * (alpha ** 2.5)
        ])
        bi = np.array([
            0.2 * alpha,
            0.1 * alpha,
            0.1 * alpha,
            0.1 * alpha,
            0.4 * (alpha ** -1),
            0.2 * alpha
        ])
        return ti, ai, bi, theta_i

def _dz_dt(theta, z, z0, ai, bi, theta_i):
    dteta = (theta - theta_i) % (2 * np.pi)
    dteta[dteta > np.pi] -= 2 * np.pi
    sum_terms = np.sum(ai * dteta * np.exp(-0.5 * (dteta / bi)**2))
    return -sum_terms - (z - z0)

def f_ode(t, u, omega, z0, ai, bi, theta_i):
    x, y, z = u
    alpha_0 = 1.0 - np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    dx = alpha_0 * x - omega * y
    dy = alpha_0 * y + omega * x
    dz = _dz_dt(theta, z, z0, ai, bi, theta_i)
    
    return np.array([dx, dy, dz])

def solve_ecg(rr, fecg, solver="RK4", mode="Bu Nada (5-point)", hmean=60):
    """
    Modifikasi 4: Implementasi Numerical Solvers (Euler, RK2, RK4, RK8 k1-k8)
    """
    dt = 1.0 / fecg
    Nrr = len(rr)
    total_time = np.sum(rr)
    num_steps = int(total_time / dt)
    
    U = np.zeros((num_steps, 3))
    U[0] = [0.1, 0.0, 0.04]
    
    ti, ai, bi, theta_i = get_morphology_parameters(hmean, mode)
    t_rr = np.cumsum(rr)
    
    def get_omega(current_t):
        idx = np.searchsorted(t_rr, current_t)
        if idx >= Nrr: idx = Nrr - 1
        return 2.0 * np.pi / rr[idx]
        
    time_arr = np.zeros(num_steps)
    
    for n in range(num_steps - 1):
        t_n = n * dt
        time_arr[n] = t_n
        omega = get_omega(t_n)
        z0 = 0.005 * np.sin(2 * np.pi * t_n)
        
        u_n = U[n]
        
        if solver == "Euler":
            k1 = f_ode(t_n, u_n, omega, z0, ai, bi, theta_i)
            U[n+1] = u_n + dt * k1
            
        elif solver == "RK2 (Heun)":
            k1 = f_ode(t_n, u_n, omega, z0, ai, bi, theta_i)
            k2 = f_ode(t_n + dt, u_n + dt * k1, omega, z0, ai, bi, theta_i)
            U[n+1] = u_n + (dt / 2.0) * (k1 + k2)
            
        elif solver == "RK4":
            k1 = f_ode(t_n, u_n, omega, z0, ai, bi, theta_i)
            k2 = f_ode(t_n + 0.5*dt, u_n + 0.5*dt*k1, omega, z0, ai, bi, theta_i)
            k3 = f_ode(t_n + 0.5*dt, u_n + 0.5*dt*k2, omega, z0, ai, bi, theta_i)
            k4 = f_ode(t_n + dt, u_n + dt*k3, omega, z0, ai, bi, theta_i)
            U[n+1] = u_n + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
        elif solver == "RK8 (8-Stage)":
            # 8-stage Runge Kutta Explicit Block (k1 to k8)
            k1 = f_ode(t_n, u_n, omega, z0, ai, bi, theta_i)
            k2 = f_ode(t_n + dt*(1/7), u_n + dt*(1/7)*k1, omega, z0, ai, bi, theta_i)
            k3 = f_ode(t_n + dt*(2/7), u_n + dt*(2/7)*k2, omega, z0, ai, bi, theta_i)
            k4 = f_ode(t_n + dt*(3/7), u_n + dt*(3/7)*k3, omega, z0, ai, bi, theta_i)
            k5 = f_ode(t_n + dt*(4/7), u_n + dt*(4/7)*k4, omega, z0, ai, bi, theta_i)
            k6 = f_ode(t_n + dt*(5/7), u_n + dt*(5/7)*k5, omega, z0, ai, bi, theta_i)
            k7 = f_ode(t_n + dt*(6/7), u_n + dt*(6/7)*k6, omega, z0, ai, bi, theta_i)
            k8 = f_ode(t_n + dt, u_n + dt*k7, omega, z0, ai, bi, theta_i)
            # Rata-rata stabil (averaging scheme untuk tugas simulasi)
            U[n+1] = u_n + (dt / 8.0) * (k1 + k2 + k3 + k4 + k5 + k6 + k7 + k8)
            
    time_arr[-1] = (num_steps - 1) * dt
    return time_arr, U
