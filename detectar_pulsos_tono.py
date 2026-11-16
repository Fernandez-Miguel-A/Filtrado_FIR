import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def detectar_pulsos_tono(x, fs, f0, bw=20, pfa=1e-6) -> list[np.ndarray[bool], np.ndarray[float], float]:
    """
    x: señal de entrada con ruido
    fs: frecuencia de muestreo (Hz)
    f0: frecuencia del tono a detectar (Hz)
    bw: ancho de banda del filtro (Hz)
    pfa: probabilidad de falsa alarma
    """

    # 1. Filtro Pasabanda (Butterworth de orden 4)
    low = (f0 - bw / 2) / (fs / 2)
    high = (f0 + bw / 2) / (fs / 2)
    b, a = signal.butter(4, [low, high], btype='band')

##
    impulso = np.zeros(31)   ## N = 31
    impulso[0] = 1.0  # El impulso vale 1 en n=0 y 0 en el resto

    # Calcular h[n] filtrando el impulso
    h = signal.lfilter(b, a, impulso)

    print(f"\n\n a: {a}, len: {len(a)},\n b: {b}, len: {len(b)} \n\n")
    print(f"\n\n h(k):", h)
##
    x_filtrada = signal.filtfilt(b, a, x)

    # 2. Extracción de envolvente vía Hilbert
    envolvente = np.abs(signal.hilbert(x_filtrada))

    # 3. Estimación del nivel de ruido y cálculo de umbral
    # (Asumiendo que los primeros momentos o valores mínimos
    # representan el piso de ruido)
    sigma_n = np.median(envolvente) / np.sqrt(np.pi / 2)
    ##umbral = sigma_n * np.sqrt(-2 * np.log(pfa))

    umbral = 0.6 * np.max(envolvente)

    # 4. Vector booleano de detección
    deteccion = envolvente >= umbral
    print("envolvente_type: ", type(envolvente), envolvente, "sigma_n", sigma_n, "umbral", umbral)
    
    return deteccion, envolvente, umbral, x_filtrada, h



if __name__ == '__main__':
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
    c = np.zeros(15, dtype=float)
    d = 4.0 * np.sin(2 * np.pi * f_pulsos * t[0:10])
    e = np.zeros(150, dtype=float)
    f = 4.0 * np.sin(2 * np.pi * f_pulsos * t[0:10])
    g = np.zeros(len(t)-65-10-15-10-150-10, dtype=float)
    componente_pulsos = np.concatenate((a, b, c, d, e, f, g))
    print(f"\n\n len(t): {len(t)}, len(componente_pulsos): {len(componente_pulsos)}")

    # Entrada: suma de ambas senoides
    entrada = componente_baja + componente_alta + componente_pulsos


    o = detectar_pulsos_tono(entrada, fs, f_pulsos)
    ##print("\n OBJ, ", o[3:4])

    salida_pulsos = o[3]
    coeficientes = o[4]
    ##print("\n\n salida: ", salida_pulsos)

    plt.plot(t*1000, entrada, "blue", t*1000, salida_pulsos, "red", t*1000, o[1], "darkorange", t*1000, o[0], "green")
    #plt.show()

## otro
    # Graficar la respuesta en frecuencia
    w_freq, H = np.abs(np.fft.rfft(coeficientes, 1024)), np.fft.rfftfreq(1024, 1/fs)

    plt.figure(figsize=(8, 4))
    plt.plot(H, 20 * np.log10(w_freq + 1e-12))
    plt.title(f"Respuesta en Frecuencia del Filtro FIR Pasa Banda\n ¡Atencion NO ES la señal filtrada propiamente!")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud (dB)")
    plt.axvline(f_pulsos, color='r', linestyle='--', label=f'f_pulsos = {f_pulsos} Hz')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
##
    input()