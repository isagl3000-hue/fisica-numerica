#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 22:22:13 2025

@author: isaias-gl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as stats

# Cargar los datos
df = pd.read_csv('MuRun2010B.csv')

# Verificar las columnas disponibles
print("Columnas en el dataset:", df.columns.tolist())

# 2(a) Calcular la masa del bosón Z usando dinámica relativista
def calcular_masa_invariante(E1, px1, py1, pz1, E2, px2, py2, pz2):
    """
    Calcula la masa invariante de un sistema de dos partículas
    Fórmula: M^2 = (E1 + E2)^2 - (px1+px2)^2 - (py1+py2)^2 - (pz1+pz2)^2
    """
    E_total = E1 + E2
    px_total = px1 + px2
    py_total = py1 + py2
    pz_total = pz1 + pz2
    
    masa_cuadrada = E_total**2 - px_total**2 - py_total**2 - pz_total**2
    # Asegurar que no haya valores negativos por errores numéricos
    masa_cuadrada = np.maximum(masa_cuadrada, 0)
    return np.sqrt(masa_cuadrada)

# Calcular masas para todas las colisiones
masas = calcular_masa_invariante(
    df['E1'], df['px1'], df['py1'], df['pz1'],
    df['E2'], df['px2'], df['py2'], df['pz2']
)

# Convertir a GeV (asumiendo que las unidades están en GeV)
masas_gev = masas

# 2(b) Histograma de frecuencias
plt.figure(figsize=(12, 5))

# Histograma normal
plt.subplot(1, 2, 1)
n, bins, patches = plt.hist(masas_gev, bins=120, range=(0, 100), 
                           alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Masa invariante (GeV)')
plt.ylabel('Frecuencia')
plt.title('Histograma de masas del bosón Z')
plt.grid(True, alpha=0.3)

# 2(c) Histograma con escala logarítmica
plt.subplot(1, 2, 2)
# Evitar log(0) reemplazando ceros con un valor pequeño
log_freq = np.log10(np.where(n == 0, 0.1, n))
plt.step(bins[:-1], log_freq, where='post', color='red', linewidth=2)
plt.xlabel('Masa invariante (GeV)')
plt.ylabel('log10(Frecuencia)')
plt.title('Histograma logarítmico de masas')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Análisis de la resonancia alrededor de 92 GeV
# Filtrar datos alrededor del pico del Z
mask_z = (masas_gev > 85) & (masas_gev < 100)
masas_z = masas_gev[mask_z]

# Ajustar una distribución gaussiana para estimar la masa y su incertidumbre
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Histograma para la región del Z
n_z, bins_z = np.histogram(masas_z, bins=60, density=False)
bin_centers_z = (bins_z[:-1] + bins_z[1:]) / 2

# Estimación inicial de parámetros
A0 = np.max(n_z)
mu0 = bin_centers_z[np.argmax(n_z)]
sigma0 = 2.0

try:
    popt, pcov = curve_fit(gauss, bin_centers_z, n_z, p0=[A0, mu0, sigma0])
    A_fit, mu_fit, sigma_fit = popt
    perr = np.sqrt(np.diag(pcov))
    
    masa_z = mu_fit
    incertidumbre_z = perr[1]  # Incertidumbre en mu
    
    print(f"\n--- RESULTADOS DEL BOSÓN Z ---")
    print(f"Masa del bosón Z: {masa_z:.3f} ± {incertidumbre_z:.3f} GeV")
    print(f"Ancho del pico (sigma): {sigma_fit:.3f} GeV")
    
    # Graficar el ajuste
    plt.figure(figsize=(10, 6))
    plt.hist(masas_z, bins=60, alpha=0.7, color='lightblue', 
             edgecolor='black', label='Datos')
    x_fit = np.linspace(85, 100, 200)
    plt.plot(x_fit, gauss(x_fit, *popt), 'r-', linewidth=2, 
             label=f'Ajuste Gaussiano\nμ = {mu_fit:.2f} ± {perr[1]:.2f} GeV')
    plt.xlabel('Masa invariante (GeV)')
    plt.ylabel('Frecuencia')
    plt.title('Ajuste Gaussiano para la masa del bosón Z')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
except Exception as e:
    print(f"Error en el ajuste gaussiano: {e}")
    # Estimación simple si falla el ajuste
    masa_z = np.mean(masas_z)
    incertidumbre_z = np.std(masas_z) / np.sqrt(len(masas_z))
    print(f"Masa del bosón Z (estimación simple): {masa_z:.3f} ± {incertidumbre_z:.3f} GeV")

# 2(d) Análisis de otras resonancias
print(f"\n--- ANÁLISIS DE OTRAS RESONANCIAS ---")

# Buscar picos en el histograma completo
from scipy.signal import find_peaks

# Suavizar el histograma para mejor detección de picos
n_smooth = np.convolve(n, np.ones(5)/5, mode='same')
peaks, properties = find_peaks(n_smooth, height=50, distance=10)

print(f"Se encontraron {len(peaks)} picos significativos en el histograma:")
for i, peak in enumerate(peaks):
    masa_pico = (bins[peak] + bins[peak+1]) / 2
    altura = n_smooth[peak]
    print(f"Pico {i+1}: Masa ≈ {masa_pico:.1f} GeV, Frecuencia = {altura:.0f} eventos")

# Verificar picos en regiones conocidas
regiones = {
    "J/ψ": (2.5, 3.5),
    "ψ(2S)": (3.5, 4.0),
    "Υ(1S)": (9.0, 10.0),
    "Z": (89, 94)
}

print(f"\nBúsqueda en regiones de resonancias conocidas:")
for particula, (min_masa, max_masa) in regiones.items():
    mask = (masas_gev > min_masa) & (masas_gev < max_masa)
    eventos = np.sum(mask)
    if eventos > 0:
        print(f"{particula}: {eventos} eventos en región {min_masa}-{max_masa} GeV")
