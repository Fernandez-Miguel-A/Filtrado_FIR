import numpy as np
from scipy.signal import butter, lfilter, hilbert, find_peaks
import matplotlib.pyplot as plt

def detectar_pulso_tono(senal, fs, f_pulsos, ancho_banda=30.0, factor_umbral=4.0):
    """
    Detecta pulsos de tono a una frecuencia f_pulsos dentro de una señal ruidosa.
    
    Parámetros:
    - senal: array 1D con la señal temporal.
    - fs: frecuencia de muestreo en Hz.
    - f_pulsos: frecuencia del tono objetivo en Hz.
    - ancho_banda: ancho de banda del filtro pasabanda en Hz.
    - factor_umbral: multiplicador de la desviación estándar para el umbral de detección.
    
    Retorna:
    - sig_filtrada: señal tras el filtro pasabanda.
    - envolvente: envolvente analítica de la señal filtrada.
    - umbral: nivel de umbral calculado.
    - indices_picos: índices temporales donde se detectaron los pulsos.
    """
    # 1. Diseño de filtro Butterworth pasabanda
    nyq = 0.5 * fs
    low = max(0.1, (f_pulsos - ancho_banda / 2.0) / nyq)
    high = min(0.99, (f_pulsos + ancho_banda / 2.0) / nyq)
    
    b, a = butter(4, [low, high], btype='band')
    
    # Filtrado de la señal
    sig_filtrada = lfilter(b, a, senal)
    
    # 2. Extracción de la envolvente mediante la Transformada de Hilbert
    senal_analitica = hilbert(sig_filtrada)
    envolvente = np.abs(senal_analitica)
    
    # Suavizado opcional de la envolvente para evitar falsos positivos por fluctuaciones del ruido
    ventana_suavizado = int(fs * 0.005)  # Ventana de 5 ms
    if ventana_suavizado < 1:
        ventana_suavizado = 1
    kernel = np.ones(ventana_suavizado) / ventana_suavizado
    envolvente_suav = np.convolve(envolvente, kernel, mode='same')
    
    # 3. Cálculo dinámico del umbral basado en el ruido de fondo
    ruido_mediano = np.median(envolvente_suav)
    ruido_std = np.std(envolvente_suav)
    umbral = ruido_mediano + factor_umbral * ruido_std
    
    # 4. Detección de picos que superan el umbral
    # 'distance' evita múltiples detecciones dentro del mismo pulso (ej. separación mínima de 20 ms)
    distancia_minima = int(fs * 0.02) 
    indices_picos, _ = find_peaks(envolvente_suav, height=umbral, distance=distancia_minima)
    
    return sig_filtrada, envolvente_suav, umbral, indices_picos

# ==========================================
# Ejemplo de simulación, ejecución y graficación
# ==========================================
if __name__ == "__main__":
    # Parámetros de la simulación
    fs = 10000.0          # Frecuencia de muestreo: 10 kHz
    duracion = 2.0        # Duración en segundos
    t = np.linspace(0, duracion, int(fs * duracion), endpoint=False)
    
    f_pulsos = 1750.0     # Frecuencia del tono a buscar: 1.75 kHz
    f_pulsos = 75.0 ## frec de 75 Hz
    
    # Generar ruido gaussiano blanco (AWGN)
    np.random.seed(42)
    senal_ruidosa = np.random.normal(0, 1.2, len(t))
    
    # Insertar pulsos de tono en intervalos específicos (ej. de 0.4s a 0.55s y de 1.1s a 1.25s)
    mascara_pulso1 = (t >= 0.4) & (t <= 0.55)
    mascara_pulso2 = (t >= 1.1) & (t <= 1.25)
    
    amplitud_pulso = 3.0
    senal_ruidosa[mascara_pulso1] += amplitud_pulso * np.sin(2 * np.pi * f_pulsos * t[mascara_pulso1])
    senal_ruidosa[mascara_pulso2] += amplitud_pulso * np.sin(2 * np.pi * f_pulsos * t[mascara_pulso2])
    
    # Ejecutar la función de detección
    sig_filt, envolvente, umbral, picos = detectar_pulso_tono(
        senal_ruidosa, fs, f_pulsos, ancho_banda=40.0, factor_umbral=4.5
    )
    
    # Graficar resultados con matplotlib
    plt.figure(figsize=(12, 9))
    
    # Subgráfico 1: Señal original con ruido
    plt.subplot(3, 1, 1)
    plt.plot(t, senal_ruidosa, color='gray', alpha=0.7, label='Señal con Ruido Gaussiano')
    plt.title('Detección de Pulsos de Tono en Ruido')
    plt.ylabel('Amplitud')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Subgráfico 2: Señal filtrada pasabanda
    plt.subplot(3, 1, 2)
    plt.plot(t, sig_filt, color='dodgerblue', alpha=0.8, label=f'Señal Filtrada (Pasabanda en {f_pulsos} Hz)')
    plt.ylabel('Amplitud')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Subgráfico 3: Envolvente, umbral y marcas de pulso detectado
    plt.subplot(3, 1, 3)
    plt.plot(t, envolvente, color='darkorange', label='Envolvente (Hilbert)')
    plt.axhline(umbral, color='red', linestyle='--', label='Umbral de Detección')
    if len(picos) > 0:
        plt.plot(t[picos], envolvente[picos], 'ro', markersize=8, label='Pulsos Detectados')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Magnitud')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()