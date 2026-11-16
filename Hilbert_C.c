#include "arm_math.h"
#include "arm_const_structs.h"

// Tamaño de la FFT (debe ser potencia de 2: 32, 64, 128, 256, 512, 1024, etc.)
#define FFT_SIZE 256

/**
 * @brief Calcula la señal analítica equivalente a scipy.signal.hilbert
 * @param input  Arreglo de entrada real de tamaño FFT_SIZE
 * @param output Arreglo de salida complejo intercalado de tamaño 2 * FFT_SIZE
 *               [Re0, Im0, Re1, Im1, ...]
 * 
 * ¡Se supone que el tamaño de la señal es PAR para que sea estable!
 */
void hilbert_cmsis_f32(const float32_t *input, float32_t *output) {
    // 1. Preparar el buffer complejo: parte real = input, parte imaginaria = 0
    for (uint32_t i = 0; i < FFT_SIZE; i++) {
        output[2 * i]     = input[i]; // Parte Real
        output[2 * i + 1] = 0.0f;     // Parte Imaginaria
    }

    // 2. Ejecutar FFT directa (in-place)
    // Nota: Usa la instancia correspondiente a tu FFT_SIZE (ej. arm_cfft_sR_f32_len256)
    arm_cfft_f32(&arm_cfft_sR_f32_len256, output, 0, 1);

    // 3. Aplicar la máscara en frecuencia (duplicar frecuencias positivas, anular negativas)
    // Índice k = 0 (DC) se mantiene igual (multiplicado por 1)
    // Índice k = FFT_SIZE/2 (Nyquist) se mantiene igual (multiplicado por 1)

    // Frecuencias positivas: k = 1 hasta (FFT_SIZE / 2) - 1 -> multiplicar por 2
    for (uint32_t k = 1; k < (FFT_SIZE / 2); k++) {
        output[2 * k]     *= 2.0f; // Real
        output[2 * k + 1] *= 2.0f; // Imaginaria
    }

    // Frecuencias negativas: k = (FFT_SIZE / 2) + 1 hasta FFT_SIZE - 1 -> multiplicar por 0
    for (uint32_t k = (FFT_SIZE / 2) + 1; k < FFT_SIZE; k++) {
        output[2 * k]     = 0.0f;  // Real
        output[2 * k + 1] = 0.0f;  // Imaginaria
    }

    // 4. Ejecutar FFT inversa (ifftFlag = 1, bitReverseFlag = 1)
    // CMSIS-DSP escala automáticamente por 1/N durante la IFFT
    arm_cfft_f32(&arm_cfft_sR_f32_len256, output, 1, 1);
}



int uso(void)
{
    float32_t envelope[FFT_SIZE];
    // Calcula la magnitud compleja |z[n]| de cada muestra
    arm_cmplx_mag_f32(output, envelope, FFT_SIZE);

}