import numpy as np
import matplotlib.pyplot as plt

def diseñar_fir_pasa_altos(fs, fc, N):
    if N % 2 == 0:
        raise ValueError("N debe ser impar.")
    
    M = N - 1
    centro = M // 2
    wc = 2 * np.pi * (fc / fs)
    
    n = np.arange(N)
    
    # 1. Respuesta ideal
#    h_ideal = np.where(
#        n == centro,
#        1.0 - (wc / np.pi),
#        -np.sin(wc * (n - centro)) / (np.pi * (n - centro))
#    )

    h_ideal = np.empty(N, dtype=float)
    centro_mask = n == centro

    h_ideal[centro_mask] = 1.0 - (wc / np.pi)
    h_ideal[~centro_mask] = (
        -np.sin(wc * (n[~centro_mask] - centro))
        / (np.pi * (n[~centro_mask] - centro))
    )
    
    # 2. Ventana de Hamming
    w = 0.54 - 0.46 * np.cos(2 * np.pi * n / M)
    
    # 3. Coeficientes finales
    h = h_ideal * w
    return h

# Ejemplo de uso:
fs = 1000.0  # Hz
fc = 200.0   # Hz
N = 31       # Mayor N = transición más directa en la frecuencia de corte

coeficientes = diseñar_fir_pasa_altos(fs, fc, N)

# Graficar la respuesta en frecuencia
w_freq, H = np.abs(np.fft.rfft(coeficientes, 1024)), np.fft.rfftfreq(1024, 1/fs)

plt.figure(figsize=(8, 4))
plt.plot(H, 20 * np.log10(w_freq + 1e-12))
plt.title(f"Respuesta en Frecuencia del Filtro FIR Pasa Altos (fc = {fc} Hz)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (dB)")
plt.axvline(fc, color='r', linestyle='--', label=f'fc = {fc} Hz')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()