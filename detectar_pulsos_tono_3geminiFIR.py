import numpy as np
from scipy.signal import firwin, lfilter, hilbert
import matplotlib.pyplot as plt

def detectar_pulso_fir(senal, fs, f_pulsos, ancho_banda=20.0, umbral_factor=0.4):
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
    nyq = fs / 2.0
    
    # 1. Definir los límites de frecuencia del filtro pasa-banda
    f_low = max(1.0, f_pulsos - (ancho_banda / 2.0))
    f_high = min(nyq - 1.0, f_pulsos + (ancho_banda / 2.0))
    
    # 2. Diseñar el filtro FIR usando la ventana de Hamming (por defecto en firwin)
    numtaps = 101  # Orden del filtro (número de coeficientes)
    coeffs_fir = firwin(numtaps, [f_low, f_high], pass_zero=False, fs=fs)
    
    # 3. Aplicar el filtro FIR a la señal ruidosa
    senal_filtrada = lfilter(coeffs_fir, 1.0, senal) ## analoga a "arm_fir_q15()"
    
    # 4. Calcular la envolvente usando la Transformada de Hilbert
    senal_analitica = hilbert(senal_filtrada)
    envolvente = np.abs(senal_analitica)
    
    # 5. Detección mediante umbral sobre la envolvente normalizada
    max_env = np.max(envolvente)

    umbral_factor = 0.6
    ##### Se asegura que el pulso esta dentro del 60% del maximo
    if max_env > 0:
        envolvente_norm = envolvente / max_env ## normaliza: rango[1.0 - 0.0]
    else:
        envolvente_norm = envolvente
        
    instantes_deteccion = envolvente_norm > umbral_factor
    
    return senal_filtrada, envolvente, instantes_deteccion

# ==========================================
# Ejemplo de uso y graficación con Matplotlib
# ==========================================
if __name__ == "__main__":
    # Parámetros de simulación
    fs = 1000.0               # Frecuencia de muestreo (1 kHz)
    duracion = 3.0            # Duración en segundos
    t = np.arange(0, duracion, 1/fs)
    f_pulsos = 60.0           # Frecuencia del pulso que queremos detectar (60 Hz)
    
    # Generar ruido gaussiano blanco
    ruido = np.random.normal(0, 1.2, len(t))
    
    # Crear una señal base con ruido y añadir pulsos en momentos específicos (ej: 1.0s-1.3s y 2.1s-2.4s)
    senal_con_ruido = ruido.copy()
    pulso_mask = ((t >= 1.0) & (t <= 1.3)) | ((t >= 2.1) & (t <= 2.4)) | ((t >= 0.2) & (t <= 0.9))
    
    # Tono sinusoidal de 60 Hz modulado durante el pulso
    tono_puro = 4.0 * np.sin(2 * np.pi * f_pulsos * t)
    senal_con_ruido[pulso_mask] += tono_puro[pulso_mask]
    
    # Ejecutar la función de detección
    senal_filt, env, detecciones = detectar_pulso_fir(senal_con_ruido, fs, f_pulsos, ancho_banda=20.0, umbral_factor=0.35)
    
    # Graficar los resultados con matplotlib.pyplot
    plt.figure(figsize=(12, 9))
    
    # Gráfica 1: Señal original con ruido
    plt.subplot(3, 1, 1)
    plt.plot(t, senal_con_ruido, color='gray', alpha=0.8, label='Señal con Ruido + Pulsos')
    plt.title('1. Señal Original de Entrada (Con Ruido Gaussiano)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Gráfica 2: Señal filtrada y su envolvente
    plt.subplot(3, 1, 2)
    plt.plot(t, senal_filt, color='blue', alpha=0.6, label='Señal Filtrada (FIR Pasa-banda)')
    plt.plot(t, env, color='darkorange', linewidth=1.5, label='Envolvente (Hilbert)')
    plt.plot(t, detecciones, color='green', linewidth=1.5, label='Señal detectada')  ## ¡Esto es lo que busco!
    plt.axhline(y=0.6 * np.max(env), color='red', linestyle='--', label='Umbral de Detección')
    plt.title('2. Filtrado FIR y Extracción de Envolvente')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Gráfica 3: Detección de los pulsos sobre la señal original
    plt.subplot(3, 1, 3)
    plt.plot(t, senal_con_ruido, color='gray', alpha=0.5, label='Señal Original')
    # Resaltar con puntos rojos donde ocurre el pulso detectado
    plt.plot(t[detecciones], senal_con_ruido[detecciones], 'ro', markersize=3, label='Pulsos Detectados')
    plt.title('3. Ubicación Temporal de los Pulsos Detectados')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()