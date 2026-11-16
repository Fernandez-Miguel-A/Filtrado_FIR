#include "detectar_pulso_fir.h"

#include "arm_math.h"
#include "Hilbert_C_q15.h"

extern const q15_t coeffs_fir[DETECTAR_PULSO_FIR_NUM_TAPS];

void detectar_pulso_fir(const q15_t *senal,
                        uint8_t *output,
                        float32_t fs,
                        float32_t f_pulsos,
                        float32_t ancho_banda,
                        float32_t umbral_factor)
{
    q15_t estado[DETECTAR_PULSO_FIR_BLOCK_SIZE +
                 DETECTAR_PULSO_FIR_NUM_TAPS - 1U];
    q15_t filtrada[DETECTAR_PULSO_FIR_BLOCK_SIZE];
    q15_t analitica[2U * DETECTAR_PULSO_FIR_BLOCK_SIZE];
    q15_t envolvente[DETECTAR_PULSO_FIR_BLOCK_SIZE];
    arm_fir_instance_q15 filtro;
    q15_t max_envolvente = 0;
    q31_t factor_q15;
    q31_t umbral_q15;

    /* Los coeficientes ya fijan el filtro; estos parámetros se conservan
       para mantener la interfaz solicitada. */
    (void)fs;
    (void)f_pulsos;
    (void)ancho_banda;

    arm_fir_init_q15(&filtro,
                     DETECTAR_PULSO_FIR_NUM_TAPS,
                     coeffs_fir,
                     estado,
                     DETECTAR_PULSO_FIR_BLOCK_SIZE);
    arm_fir_q15(&filtro, senal, filtrada, DETECTAR_PULSO_FIR_BLOCK_SIZE);

    hilbert_cmsis_q15(filtrada, analitica);
    arm_cmplx_mag_q15(analitica, envolvente, DETECTAR_PULSO_FIR_BLOCK_SIZE);

    for (uint32_t i = 0; i < DETECTAR_PULSO_FIR_BLOCK_SIZE; i++) {
        if (envolvente[i] > max_envolvente) {
            max_envolvente = envolvente[i];
        }
    }

    factor_q15 = (q31_t)(umbral_factor * 32768.0f + 0.5f);
    umbral_q15 = ((q31_t)max_envolvente * factor_q15 + 16384) >> 15;

    for (uint32_t i = 0; i < DETECTAR_PULSO_FIR_BLOCK_SIZE; i++) {
        output[i] = (envolvente[i] > umbral_q15) ? 1U : 0U;
    }
}
