#ifndef DETECTAR_PULSO_FIR_H
#define DETECTAR_PULSO_FIR_H

#include "arm_math.h"

#include <stdint.h>

#define DETECTAR_PULSO_FIR_BLOCK_SIZE 256U
#define DETECTAR_PULSO_FIR_NUM_TAPS 101U
#define DETECTAR_PULSO_FIR_DEFAULT_ANCHO_BANDA 20.0f
#define DETECTAR_PULSO_FIR_DEFAULT_UMBRAL_FACTOR 0.4f

/*
 * Procesa un bloque de 256 muestras Q15. output contiene 256 valores:
 * 1 si la envolvente supera el umbral relativo, 0 en caso contrario.
 * En C los argumentos por defecto no existen: pasar 20.0f y 0.4f
 * para ancho_banda y umbral_factor cuando se quieran usar esos valores.
 *
 * coeffs_fir debe ser un arreglo Q15 constante de 101 coeficientes,
 * definido por el proyecto. umbral_factor debe estar entre 0 y 1.
 */
void detectar_pulso_fir(const q15_t *senal,
                        uint8_t *output,
                        float32_t fs,
                        float32_t f_pulsos,
                        float32_t ancho_banda,
                        float32_t umbral_factor);

#endif
