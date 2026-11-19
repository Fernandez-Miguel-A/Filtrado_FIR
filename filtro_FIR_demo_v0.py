import numpy as np
import matplotlib.pyplot as plt

def diseñar_fir_pasa_altos(fs, fc, N):
    if N % 2 == 0:
        raise ValueError("El número de taps N debe ser impar.")
    
    M = N - 1
    centro = M // 2
    wc = 2 * np.pi * (fc / fs)
    n = np.arange(N)
    
    # Evitar división por cero en el índice central
    denom = np.pi * (n - centro)
    denom[centro] = 1.0
    
    # Respuesta al impulso ideal
    h_ideal = -np.sin(wc * (n - centro)) / denom
    h_ideal[centro] = 1.0 - (wc / np.pi)
    
    # Ventana de Hamming
    w = 0.54 - 0.46 * np.cos(2 * np.pi * n / M)
    
    return h_ideal * w


def diseñar_fir_pasa_bajos(fs, fc, N):
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("La frecuencia de muestreo fs debe ser positiva.")
    if not np.isfinite(fc) or not 0 < fc < fs / 2:
        raise ValueError("La frecuencia de corte fc debe estar entre 0 y fs/2.")
    if not isinstance(N, (int, np.integer)) or N < 3 or N % 2 == 0:
        raise ValueError("El número de taps N debe ser impar y mayor o igual que 3.")
    
    M = N - 1
    centro = M // 2
    wc = 2 * np.pi * (fc / fs)
    n = np.arange(N)
    
    denom = np.pi * (n - centro)
    denom[centro] = 1.0
    
    h_ideal = np.sin(wc * (n - centro)) / denom
    h_ideal[centro] = wc / np.pi
    
    w = 0.54 - 0.46 * np.cos(2 * np.pi * n / M)
    
    return h_ideal * w

# 1. Parámetros del filtro
fs = 1000.0   # Frecuencia de muestreo (Hz)
fc = 200.0    # Frecuencia de corte (Hz)
N = 15        # Número de taps (impar)

# 2. Generar señal en el tiempo (0.1 segundos)
t = np.linspace(0, 0.1, int(fs * 0.1), endpoint=False)
f_baja = 50.0   # Componente de 50 Hz (debe eliminarse)
f_alta = 300.0  # Componente de 300 Hz (debe pasar)

componente_baja = np.sin(2 * np.pi * f_baja * t)
componente_alta = np.sin(2 * np.pi * f_alta * t)

# Entrada: suma de ambas senoides
entrada = componente_baja + componente_alta

# 3. Filtrar aplicando convolución
h_high = diseñar_fir_pasa_altos(fs, fc, N)
salida_high = np.convolve(entrada, h_high, mode='same')

h_low = diseñar_fir_pasa_bajos(fs, fc, N)
salida_low = np.convolve(entrada, h_low, mode='same')

# 4. Graficación comparativa
plt.figure(figsize=(10, 7))

# Gráfica 1: Entrada mezclada
plt.subplot(4, 1, 1)
plt.plot(t * 1000, entrada, color='crimson', label='Entrada (50 Hz + 300 Hz)')
plt.title('1. Señal de Entrada Mezclada (Se observan las ondulaciones de baja frecuencia)')
plt.ylabel('Amplitud')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

# Gráfica 2: Referencia de alta frecuencia pura
plt.subplot(4, 1, 2)
plt.plot(t * 1000, componente_alta, color='darkgreen', linestyle='--', label='Componente pura de 300 Hz')
plt.title('2. Señal Objetivo Ideal (Solo 300 Hz)')
plt.ylabel('Amplitud')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

# Gráfica 3: Salida filtrada filtrada pasa altos
plt.subplot(4, 1, 3)
plt.plot(t * 1000, salida_high, color='royalblue', label='Salida Filtrada (FIR Pasa Altos)')
plt.title('3. Señal Salida Filtrada (Oscilaciones lentas eliminadas)')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

# Gráfica 4: Salida filtrada pasa bajos
plt.subplot(4, 1, 4)
plt.plot(t * 1000, salida_low, color='red', label='Salida Filtrada (FIR Pasa Bajos)')
plt.title('3. Señal Salida Filtrada (Oscilaciones lentas eliminadas)')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
