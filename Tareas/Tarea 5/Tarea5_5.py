#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 22:01:58 2025

@author: isaias-gl
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import constants

# Datos del COBE - Radiación cósmica de fondo
# ν en cm⁻¹, I(ν,T) en MJy/sr, Error en kJy/sr
nu_cm = np.array([2.27, 2.72, 3.18, 3.63, 4.08, 4.54, 4.99, 5.45, 5.90, 
                  6.35, 6.81, 7.26, 7.71, 8.17, 8.62, 9.08, 9.53, 9.98,
                  10.44, 10.89, 11.34, 11.80, 12.25, 12.71, 13.16, 13.61, 
                  14.07, 14.52, 14.97, 15.43, 15.88, 16.34, 16.79, 17.24, 
                  17.70, 18.15, 18.61, 19.06, 19.51, 19.97, 20.42, 20.87, 21.33])

I_MJy = np.array([200.723, 249.508, 293.024, 327.770, 354.081, 372.079, 381.493, 
                  383.478, 378.901, 368.833, 354.063, 336.278, 316.076, 293.924, 
                  271.432, 248.239, 225.940, 204.327, 183.262, 163.830, 145.750, 
                  128.835, 113.568, 99.451, 87.036, 75.876, 65.766, 57.008, 
                  49.223, 42.267, 36.352, 31.062, 26.580, 22.644, 19.255, 
                  16.391, 13.811, 11.716, 9.921, 8.364, 7.087, 5.801, 4.523])

error_kJy = np.array([14, 19, 25, 23, 22, 21, 18, 18, 16, 14, 13, 12, 11, 10, 
                      11, 12, 14, 16, 18, 22, 22, 23, 23, 23, 22, 21, 20, 19, 
                      19, 19, 21, 23, 26, 28, 30, 32, 33, 35, 41, 55, 88, 155, 282])

# Convertir unidades
# 1 cm⁻¹ = 3e10 Hz, 1 MJy/sr = 1e-20 W/m²/Hz/sr
nu_Hz = nu_cm * 3e10  # Convertir a Hz
I_Wm2 = I_MJy * 1e-20  # Convertir a W/m²/Hz/sr
error_Wm2 = error_kJy * 1e-23  # Convertir error a W/m²/Hz/sr

# Función de Planck para ajuste
def planck_spectrum(nu, T):
    """
    Ley de Planck para la radiación de cuerpo negro
    I(ν,T) = (2hν³/c²) / (exp(hν/kT) - 1)
    """
    h = constants.h  # 6.626e-34 J·s
    c = constants.c  # 3e8 m/s
    k = constants.k  # 1.381e-23 J/K
    
    prefactor = (2 * h * nu**3) / (c**2)
    exponent = (h * nu) / (k * T)
    
    return prefactor / (np.exp(exponent) - 1)

# Versión para curve_fit (con factor de escala)
def planck_fit(nu, T, scale_factor):
    """Planck con factor de escala para ajuste"""
    return scale_factor * planck_spectrum(nu, T)

print("=== AJUSTE DEL ESPECTRO DE CUERPO NEGRO - DATOS COBE ===")

# Guess inicial
T_guess = 2.7  # K (valor esperado para CMB)
scale_guess = 1e20  # Factor de escala aproximado

# Ajuste por mínimos cuadrados
p0 = [T_guess, scale_guess]
popt, pcov = curve_fit(planck_fit, nu_Hz, I_Wm2, 
                       sigma=error_Wm2, absolute_sigma=True,
                       p0=p0, maxfev=5000)

T_opt, scale_opt = popt
T_err, scale_err = np.sqrt(np.diag(pcov))

print(f"\n--- RESULTADOS DEL AJUSTE ---")
print(f"Temperatura de la CMB: T = ({T_opt:.4f} ± {T_err:.4f}) K")
print(f"Factor de escala: {scale_opt:.2e} ± {scale_err:.2e}")
print(f"Temperatura en °C: {T_opt - 273.15:.2f} °C")

# Calcular χ²
I_pred = planck_fit(nu_Hz, T_opt, scale_opt)
residuals = (I_Wm2 - I_pred) / error_Wm2
chi2 = np.sum(residuals**2)
dof = len(nu_Hz) - len(popt)
chi2_reduced = chi2 / dof

print(f"\n--- ANÁLISIS DE BONDAD DE AJUSTE ---")
print(f"χ² = {chi2:.2f}")
print(f"Grados de libertad = {dof}")
print(f"χ² reducido = {chi2_reduced:.2f}")

# Gráfica de los datos y ajuste (Parte a)
plt.figure(figsize=(15, 5))

# Gráfica 1: Datos y ajuste en unidades originales
plt.subplot(1, 2, 1)
plt.errorbar(nu_cm, I_MJy, yerr=error_kJy/1000, fmt='o', 
             markersize=4, capsize=3, alpha=0.7, label='Datos COBE')

# Curva teórica
nu_fit = np.linspace(2, 22, 1000)
nu_fit_Hz = nu_fit * 3e10
I_fit_MJy = planck_fit(nu_fit_Hz, T_opt, scale_opt) * 1e20  # Volver a MJy/sr

plt.plot(nu_fit, I_fit_MJy, 'r-', linewidth=2, 
         label=f'Ajuste Planck: T = {T_opt:.3f} K')

plt.xlabel('Frecuencia (cm⁻¹)')
plt.ylabel('I(ν,T) (MJy/sr)')
plt.title('Espectro de Cuerpo Negro - Datos COBE')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfica 2: Comparación con forma teórica esperada
plt.subplot(1, 2, 2)
# Normalizar para comparar formas
I_data_norm = I_MJy / np.max(I_MJy)
I_fit_norm = I_fit_MJy / np.max(I_fit_MJy)

plt.errorbar(nu_cm, I_data_norm, yerr=error_kJy/(1000*np.max(I_MJy)), 
             fmt='o', markersize=4, capsize=3, alpha=0.7, label='Datos (normalizados)')
plt.plot(nu_fit, I_fit_norm, 'r-', linewidth=2, label='Planck (normalizado)')

plt.xlabel('Frecuencia (cm⁻¹)')
plt.ylabel('Intensidad (normalizada)')
plt.title('Comparación de Formas - Normalizado')
plt.legend()
plt.grid(True, alpha=0.3)


plt.tight_layout()
plt.show()

# Análisis de la forma característica
print(f"\n--- ANÁLISIS DE LA FORMA DEL ESPECTRO ---")
# Encontrar el pico del espectro (Ley de Wien)
peak_idx = np.argmax(I_fit_MJy)
nu_peak = nu_fit[peak_idx]
print(f"Pico del espectro: ν = {nu_peak:.1f} cm⁻¹")

# Verificar ley de Wien: ν_max ∝ T
wien_constant = 5.879e10  # Hz/K (Ley de Wien)
nu_max_expected = wien_constant * T_opt / 3e10  # Convertir a cm⁻¹
print(f"Pico esperado (Ley de Wien): ν = {nu_max_expected:.1f} cm⁻¹")
print(f"Diferencia: {abs(nu_peak - nu_max_expected):.1f} cm⁻¹")

# Comparación con valor conocido de la CMB
T_CMB_known = 2.725  # K (valor aceptado)
print(f"\n--- COMPARACIÓN CON VALOR ACEPTADO ---")
print(f"Temperatura ajustada: {T_opt:.4f} K")
print(f"Temperatura aceptada: {T_CMB_known:.4f} K")
print(f"Diferencia: {abs(T_opt - T_CMB_known):.4f} K")
print(f"Error relativo: {abs(T_opt - T_CMB_known)/T_CMB_known*100:.2f}%")

# Gráfica adicional: espectro en escala log-log
plt.figure(figsize=(10, 6))
plt.errorbar(nu_cm, I_MJy, yerr=error_kJy/1000, fmt='o', 
             capsize=3, alpha=0.7, label='Datos COBE')
plt.plot(nu_fit, I_fit_MJy, 'r-', linewidth=2, 
         label=f'Ley de Planck - T = {T_opt:.3f} K')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frecuencia (cm⁻¹) - Escala Log')
plt.ylabel('I(ν,T) (MJy/sr) - Escala Log')
plt.title('Espectro de Cuerpo Negro - Escala Log-Log')
plt.legend()
plt.grid(True, alpha=0.3, which='both')
plt.show()

print(f"\n=== CONCLUSIÓN ===")
print(f"Los datos del COBE siguen perfectamente un espectro de cuerpo negro")
print(f"a temperatura T = {T_opt:.3f} K, confirmando la naturaleza térmica")
print(f"de la Radiación Cósmica de Fondo.")