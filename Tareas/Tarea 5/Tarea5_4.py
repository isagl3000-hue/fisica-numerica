#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 21:18:25 2025

@author: isaias-gl
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as stats

# Datos del experimento RL
time_ns = np.array([0.0, 32.8, 65.6, 98.4, 131.2, 164.0, 196.8, 229.6, 262.4, 
                    295.2, 328.0, 360.8, 393.6, 426.4, 459.2, 492.0])

voltage = np.array([5.08e+00, 3.29e+00, 2.23e+00, 1.48e+00, 1.11e+00, 6.44e-01,
                   4.76e-01, 2.73e-01, 1.88e-01, 1.41e-01, 9.42e-02, 7.68e-02,
                   3.22e-02, 3.22e-02, 1.98e-02, 1.98e-02])

uncertainty = np.array([1.12e-01, 9.04e-02, 7.43e-02, 6.05e-02, 5.25e-02, 4.00e-02,
                       3.43e-02, 2.60e-02, 2.16e-02, 1.87e-02, 1.53e-02, 1.38e-02,
                       8.94e-03, 8.94e-03, 7.01e-03, 7.01e-03])

# Convertir tiempo a segundos (1 ns = 1e-9 s)
time_s = time_ns * 1e-9

# Función del modelo exponencial
def exponential_decay(t, V0, Gamma):
    """Modelo de decaimiento exponencial V(t) = V0 * exp(-Gamma * t)"""
    return V0 * np.exp(-Gamma * t)

# Ajuste por mínimos cuadrados ponderados
print("=== AJUSTE EXPONENCIAL - CIRCUITO RL ===")

# Realizar el ajuste
popt, pcov = curve_fit(exponential_decay, time_s, voltage, 
                       sigma=uncertainty, absolute_sigma=True,
                       p0=[5.0, 1e7])  # Guess inicial: V0=5V, Gamma=1e7 s⁻¹

# Extraer parámetros y errores
V0_opt, Gamma_opt = popt
V0_err, Gamma_err = np.sqrt(np.diag(pcov))

print(f"\n--- PARÁMETROS AJUSTADOS ---")
print(f"V₀ = ({V0_opt:.4f} ± {V0_err:.4f}) V")
print(f"Γ = ({Gamma_opt:.2e} ± {Gamma_err:.2e}) s⁻¹")
print(f"Γ = ({Gamma_opt/1e6:.2f} ± {Gamma_err/1e6:.2f}) × 10⁶ s⁻¹")

# Calcular χ²
voltage_pred = exponential_decay(time_s, V0_opt, Gamma_opt)
residuals = (voltage - voltage_pred) / uncertainty
chi2 = np.sum(residuals**2)
dof = len(voltage) - 2  # grados de libertad = n puntos - n parámetros
chi2_reduced = chi2 / dof

print(f"\n--- ANÁLISIS DE χ² ---")
print(f"χ² = {chi2:.2f}")
print(f"Grados de libertad = {dof}")
print(f"χ² reducido = {chi2_reduced:.2f}")

# Valor p del ajuste
p_value = 1 - stats.chi2.cdf(chi2, dof)
print(f"Valor p = {p_value:.3f}")

# Interpretación del χ²
if chi2_reduced < 1:
    chi2_interpretation = "Excelente ajuste (χ² < 1)"
elif chi2_reduced < 2:
    chi2_interpretation = "Buen ajuste (χ² < 2)"
elif chi2_reduced < 3:
    chi2_interpretation = "Ajuste aceptable (χ² < 3)"
else:
    chi2_interpretation = "Ajuste pobre (χ² ≥ 3)"

print(f"Interpretación: {chi2_interpretation}")

# Gráfica semi-log (parte a y c)
plt.figure(figsize=(15, 5))

# Gráfica lineal
plt.subplot(1, 2, 1)
plt.errorbar(time_ns, voltage, yerr=uncertainty, fmt='o', 
             capsize=3, label='Datos experimentales', alpha=0.7)
t_fit = np.linspace(0, 500, 1000)
V_fit = exponential_decay(t_fit * 1e-9, V0_opt, Gamma_opt)
plt.plot(t_fit, V_fit, 'r-', label=f'Ajuste: $V_0$ = {V0_opt:.2f} V, $\Gamma$ = {Gamma_opt/1e6:.1f}×10⁶ s⁻¹')
plt.xlabel('Tiempo (ns)')
plt.ylabel('Voltaje (V)')
plt.title('Decaimiento Exponencial - Escala Lineal')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfica semi-log (PARTE c)
plt.subplot(1, 2, 2)
plt.errorbar(time_ns, voltage, yerr=uncertainty, fmt='o', 
             capsize=3, label='Datos experimentales', alpha=0.7)
plt.semilogy(t_fit, V_fit, 'r-', label='Ajuste exponencial')
plt.xlabel('Tiempo (ns)')
plt.ylabel('Voltaje (V) - Escala Log')
plt.title('Gráfica Semi-Log (Parte c)')
plt.legend()
plt.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.show()


# Verificación física
print(f"\n--- VERIFICACIÓN FÍSICA ---")
print(f"Voltaje inicial medido: {voltage[0]:.2f} V")
print(f"Voltaje inicial ajustado: {V0_opt:.2f} V")
print(f"Voltaje a t = 200 ns (medido): {voltage[6]:.3f} V")
print(f"Voltaje a t = 200 ns (modelo): {exponential_decay(200e-9, V0_opt, Gamma_opt):.3f} V")