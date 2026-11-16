#include "Hilbert_C_q15.h"
#include "arm_const_structs.h"

#define FFT_SIZE 256

static q15_t saturate_q15(q31_t value)
{
    if (value > 32767) {
        return 32767;
    }
    if (value < -32768) {
        return -32768;
    }
    return (q15_t)value;
}

/**
 * @brief Calcula la señal analítica equivalente a scipy.signal.hilbert.
 * @param input  Arreglo de entrada real Q15 de tamaño FFT_SIZE.
 * @param output Arreglo complejo intercalado Q15 de tamaño 2 * FFT_SIZE:
 *               [Re0, Im0, Re1, Im1, ...].
 */
void hilbert_cmsis_q15(const q15_t *input, q15_t *output)
{
    for (uint32_t i = 0; i < FFT_SIZE; i++) {
        output[2 * i] = input[i];
        output[2 * i + 1] = 0;
    }

    arm_cfft_q15(&arm_cfft_sR_q15_len256, output, 0, 1);

    for (uint32_t k = 1; k < (FFT_SIZE / 2); k++) {
        output[2 * k] = saturate_q15((q31_t)output[2 * k] * 2);
        output[2 * k + 1] = saturate_q15((q31_t)output[2 * k + 1] * 2);
    }

    for (uint32_t k = (FFT_SIZE / 2) + 1; k < FFT_SIZE; k++) {
        output[2 * k] = 0;
        output[2 * k + 1] = 0;
    }

    arm_cfft_q15(&arm_cfft_sR_q15_len256, output, 1, 1);

    // Q15 FFT/IFFT escalan cada transformada por 1/FFT_SIZE.
    for (uint32_t i = 0; i < 2 * FFT_SIZE; i++) {
        output[i] = saturate_q15((q31_t)output[i] * FFT_SIZE);
    }
}
