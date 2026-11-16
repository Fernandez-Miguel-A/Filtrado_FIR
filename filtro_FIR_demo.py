import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, hilbert



def detectar_pulso_fir(senal, umbral_factor=0.6):
    """
    Detecta pulsos de una frecuencia específica usando un filtro FIR y la envolvente de Hilbert.
    
    Parámetros:
    - senal: numpy array con la señal ruidosa de entrada.
    - fs: Frecuencia de muestreo en Hz.
    - f_pulsos: Frecuencia central del tono a detectar en Hz.
    - ancho_banda: Ancho de banda total del filtro pasa-banda en Hz.
    - umbral_factor: Factor multiplicativo para definir el umbral de detección (0 a 1).
    
    Retorna:
    - senal_filtrada: Señal resultante del filtro FIR.
    - envolvente: Envolvente analítica de la señal filtrada.
    - instantes_deteccion: Máscara booleana o índices donde se detectó el pulso.
    """
    # 3. Aplicar el filtro FIR a la señal ruidosa
    ##senal_filtrada = lfilter(coeffs_fir, 1.0, senal) ## analoga a "arm_fir_q15()"

    ###salida_band = np.convolve(senal, h_band, mode='same')
    senal_filtrada = senal
    # 4. Calcular la envolvente usando la Transformada de Hilbert
    senal_analitica = hilbert(senal_filtrada)
    envolvente = np.abs(senal_analitica)
    
    # 5. Detección mediante umbral sobre la envolvente normalizada
    max_env = np.max(envolvente)


    ##### Se asegura que el pulso esta dentro del 60% del maximo
    if max_env > 0:
        envolvente_norm = envolvente / max_env ## normaliza: rango[1.0 - 0.0]
    else:
        envolvente_norm = envolvente
        
    instantes_deteccion = envolvente_norm > umbral_factor
    
    return senal_filtrada, envolvente, instantes_deteccion


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


def diseñar_fir_pasa_banda(fs, fc_low, fc_high, N):
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("La frecuencia de muestreo fs debe ser positiva.")
    if (
        not np.isfinite(fc_low)
        or not np.isfinite(fc_high)
        or not 0 < fc_low < fc_high < fs / 2
    ):
        raise ValueError(
            "Las frecuencias de corte deben cumplir 0 < fc_low < fc_high < fs/2."
        )
    if not isinstance(N, (int, np.integer)) or N < 3 or N % 2 == 0:
        raise ValueError("El número de taps N debe ser impar y mayor o igual que 3.")
    
    M = N - 1
    centro = M // 2
    wc_low = 2 * np.pi * (fc_low / fs)
    wc_high = 2 * np.pi * (fc_high / fs)
    n = np.arange(N)
    desplazamiento = n - centro
    
    denom = np.pi * desplazamiento
    denom[centro] = 1.0
    
    h_ideal = (
        np.sin(wc_high * desplazamiento) - np.sin(wc_low * desplazamiento)
    ) / denom
    h_ideal[centro] = (wc_high - wc_low) / np.pi
    
    w = 0.54 - 0.46 * np.cos(2 * np.pi * n / M)
    
    return h_ideal * w

# 1. Parámetros del filtro
fs = 1000.0   # Frecuencia de muestreo (Hz)
fc = 200.0    # Frecuencia de corte (Hz)
N = 31        # Número de taps (impar)

# 2. Generar señal en el tiempo (0.1 segundos)
t = np.linspace(0, 0.1, int(fs * 0.1*10), endpoint=False)
f_baja = 50.0   # Componente de 50 Hz (debe eliminarse)
f_alta = 300.0  # Componente de 300 Hz (debe pasar)
f_pulsos = 75.0

componente_baja = np.sin(2 * np.pi * f_baja * t)
componente_alta = np.sin(2 * np.pi * f_alta * t)
a = np.zeros(65, dtype=float)
b = 4.0 * np.sin(2 * np.pi * f_pulsos * t[0:10])
c = np.zeros(15*1, dtype=float)
d = 4.0 * np.sin(2 * np.pi * f_pulsos * t[0:10])
e = np.zeros(150, dtype=float)
f = 4.0 * np.sin(2 * np.pi * f_pulsos * t[0:10])
g = np.zeros(len(t)-65-10-15*1-10-150-10, dtype=float)
componente_pulsos = np.concatenate((a, b, c, d, e, f, g))

# Entrada: suma de ambas senoides
entrada = componente_baja + componente_alta + componente_pulsos

# 3. Filtrar aplicando convolución
h_high = diseñar_fir_pasa_altos(fs, fc, N)
salida_high = np.convolve(entrada, h_high, mode='same')

h_low = diseñar_fir_pasa_bajos(fs, fc, N)
salida_low = np.convolve(entrada, h_low, mode='same')

h_band = diseñar_fir_pasa_banda(fs, 75-10, 75+10, N)
salida_band = np.convolve(entrada, h_band, mode='same')
senal_filt, env, detecciones = detectar_pulso_fir(salida_band, h_band)

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

plt.figure(figsize=(10, 7))
plt.plot(t * 1000, componente_pulsos, color='blue', linestyle='--', label=f'Componente pura de {f_pulsos} Hz')
plt.plot(t*1000, salida_band, "orange", label=f'Señal filtrada')
plt.plot(t*1000, detecciones, "green", label=f'Deteccion de pulsos de {f_pulsos} Hz')
plt.ylabel('Amplitud')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
