#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 10:52:05 2025

@author: isaias-gl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Configuración de matplotlib para mejor visualización
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# Cargar los datos
print("Cargando datos...")
df = pd.read_csv('Jpsimumu_Run2011A.csv')

# Mostrar información básica del dataset
print(f"Número de eventos: {len(df)}")
print("\nPrimeras filas del dataset:")
print(df.head())
print("\nEstadísticas descriptivas de las masas (si ya existen):")

# =============================================================================
# PARTE 1(a): Cálculo de la masa invariante
# =============================================================================
print("\n" + "="*60)
print("CALCULANDO MASAS INVARIANTES")
print("="*60)

# Calcular la masa invariante para cada evento
# M = √[(E1 + E2)² - (px1 + px2)² - (py1 + py2)² - (pz1 + pz2)²]

# Suma de energías y momentos
E_total = df['E1'] + df['E2']
px_total = df['px1'] + df['px2']
py_total = df['py1'] + df['py2']
pz_total = df['pz1'] + df['pz2']

# Calcular masa invariante
masas = np.sqrt(E_total**2 - px_total**2 - py_total**2 - pz_total**2)

# Convertir a GeV (si los datos están en otra unidad, ajustar)
# Asumiendo que los datos ya están en GeV
df['Masa'] = masas

# Mostrar estadísticas reales de los datos
print(f"Masa mínima: {df['Masa'].min():.3f} GeV")
print(f"Masa máxima: {df['Masa'].max():.3f} GeV")
print(f"Masa promedio: {df['Masa'].mean():.3f} GeV")
print(f"Desviación estándar: {df['Masa'].std():.3f} GeV")

# =============================================================================
# PARTE 1(b): Histograma de frecuencias
# =============================================================================
print("\n" + "="*60)
print("CREANDO HISTOGRAMA")
print("="*60)

# Usar el rango real de los datos con un pequeño margen
masa_min = max(0, df['Masa'].min() - 0.1)
masa_max = df['Masa'].max() + 0.1

n_bins = 100  # Reducido a 100 bins como se solicita
hist, bins, patches = plt.hist(df['Masa'], bins=n_bins, 
                              range=(masa_min, masa_max),
                              alpha=0.7, color='blue', 
                              edgecolor='black', linewidth=0.5)

plt.xlabel('Masa Invariante (GeV/c²)', fontsize=14)
plt.ylabel('Número de Eventos', fontsize=14)
plt.title('Espectro de Masa Dimuónica - Partículas que Decaen a dos Muones', fontsize=16)
plt.grid(True, alpha=0.3)

# Mejorar los ejes
plt.xlim(masa_min, masa_max)

plt.tight_layout()
plt.show()

# =============================================================================
# Función para ajuste gaussiano (para cálculo de incertidumbres)
# =============================================================================
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# =============================================================================
# PARTE 1(c): Análisis de las resonancias con cálculo de incertidumbres
# =============================================================================
print("\n" + "="*60)
print("ANÁLISIS DE RESONANCIAS CON INCERTIDUMBRES")
print("="*60)

# Encontrar los picos en el histograma
hist_suave = gaussian_filter1d(hist, sigma=1.5)

# Ajustar parámetros para encontrar exactamente 2 picos principales
picos, propiedades = find_peaks(hist_suave, height=30, distance=15, prominence=10)

print(f"Número de picos encontrados: {len(picos)}")

# Mostrar posiciones y alturas de los picos
print("\nPicos encontrados (masa en GeV):")
for i, pico in enumerate(picos):
    masa_pico = bins[pico] + (bins[1] - bins[0])/2  # Centro del bin
    altura = hist_suave[pico]
    print(f"Pico {i+1}: {masa_pico:.3f} GeV (altura: {altura:.0f} eventos)")

# Si encontramos más de 2 picos, tomar solo los 2 más prominentes
if len(picos) > 2:
    print("\nSe encontraron más de 2 picos. Tomando los 2 más prominentes...")
    # Ordenar por prominencia (altura)
    prominencias = propiedades['prominences']
    indices_ordenados = np.argsort(prominencias)[::-1]  # Orden descendente
    picos = picos[indices_ordenados[:2]]
    print(f"Picos seleccionados: {[bins[p] + (bins[1]-bins[0])/2 for p in picos]}")

# =============================================================================
# CÁLCULO DE INCERTIDUMBRES PARA CADA PICO
# =============================================================================
print("\n" + "="*60)
print("CÁLCULO DE INCERTIDUMBRES")
print("="*60)

# Partículas conocidas en el rango observado (2-5 GeV)
particulas_conocidas = {
    3.096: "J/ψ (charmonio)",
    3.686: "ψ(2S)"
}

# Para cada pico, hacer un ajuste gaussiano para determinar la incertidumbre
masas_picos = []
incertidumbres = []
identificaciones = []

for i, pico in enumerate(picos):
    masa_aproximada = bins[pico] + (bins[1] - bins[0])/2
    
    # Definir región alrededor del pico para el ajuste (±2σ aproximado)
    ancho_ventana = 0.3  # GeV
    mask = (bins[:-1] >= masa_aproximada - ancho_ventana) & (bins[:-1] <= masa_aproximada + ancho_ventana)
    
    x_fit = bins[:-1][mask] + (bins[1] - bins[0])/2  # Puntos centrales de los bins
    y_fit = hist[mask]
    
    try:
        # Estimaciones iniciales para el ajuste
        A0 = hist_suave[pico]  # Altura
        mu0 = masa_aproximada  # Posición
        sigma0 = 0.05  # Ancho inicial estimado
        
        # Realizar ajuste gaussiano
        popt, pcov = curve_fit(gauss, x_fit, y_fit, p0=[A0, mu0, sigma0])
        
        # Extraer parámetros e incertidumbre
        A, mu, sigma = popt
        incertidumbre_mu = np.sqrt(pcov[1, 1])  # Incertidumbre en la posición
        
        masas_picos.append(mu)
        incertidumbres.append(incertidumbre_mu)
        
        # Identificar la partícula
        particula_identificada = "Desconocida"
        for masa_ref, nombre in particulas_conocidas.items():
            if abs(mu - masa_ref) < 0.1:  # Tolerancia de 100 MeV
                particula_identificada = nombre
                break
        
        identificaciones.append(particula_identificada)
        
        print(f"\nPico {i+1}:")
        print(f"  Masa: {mu:.3f} ± {incertidumbre_mu:.3f} GeV")
        print(f"  Ancho (σ): {sigma:.3f} GeV")
        print(f"  Identificación: {particula_identificada}")
        
    except Exception as e:
        print(f"Error en el ajuste del pico {i+1}: {e}")
        # Usar estimación simple si falla el ajuste
        ancho_bin = bins[1] - bins[0]
        incertidumbre_simple = ancho_bin / np.sqrt(hist_suave[pico])
        masas_picos.append(masa_aproximada)
        incertidumbres.append(incertidumbre_simple)
        
        # Identificación simple
        particula_identificada = "Desconocida"
        for masa_ref, nombre in particulas_conocidas.items():
            if abs(masa_aproximada - masa_ref) < 0.1:
                particula_identificada = nombre
                break
        identificaciones.append(particula_identificada)
        
        print(f"\nPico {i+1} (estimación simple):")
        print(f"  Masa: {masa_aproximada:.3f} ± {incertidumbre_simple:.3f} GeV")
        print(f"  Identificación: {particula_identificada}")

# =============================================================================
# Histograma final con picos e incertidumbres marcados
# =============================================================================
plt.figure(figsize=(14, 8))

# Histograma principal
n, bins, patches = plt.hist(df['Masa'], bins=n_bins, 
                           range=(masa_min, masa_max),
                           alpha=0.7, color='lightblue', 
                           edgecolor='gray', linewidth=0.5)

# Marcar los picos encontrados con sus incertidumbres
colors = ['red', 'green', 'orange', 'purple']
for i, (pico, masa, incert, ident) in enumerate(zip(picos, masas_picos, incertidumbres, identificaciones)):
    color = colors[i % len(colors)]
    
    # Línea vertical en la posición del pico
    plt.axvline(x=masa, color=color, linestyle='-', alpha=0.8, linewidth=2)
    
    # Banda de incertidumbre
    plt.axvspan(masa - incert, masa + incert, alpha=0.2, color=color)
    
    # Texto con la información
    plt.text(masa, hist_suave[pico] + 15, 
             f'{ident}\n{masa:.3f} ± {incert:.3f} GeV', 
             rotation=0, va='bottom', ha='center', 
             fontweight='bold', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.2))

plt.xlabel('Masa Invariante (GeV/c²)', fontsize=14)
plt.ylabel('Número de Eventos', fontsize=14)
plt.title('Espectro de Masa Dimuónica - Picos de Resonancia con Incertidumbres', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xlim(masa_min, masa_max)

plt.tight_layout()
plt.savefig('espectro_masas_incertidumbres.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "="*60)
print("RESUMEN FINAL")
print("="*60)

print(f"\nRango de masas observado: {df['Masa'].min():.3f} - {df['Masa'].max():.3f} GeV")
print(f"Número total de eventos: {len(df)}")
print(f"Número de picos identificados: {len(masas_picos)}")

print("\nResultados de las resonancias:")
for i, (masa, incert, ident) in enumerate(zip(masas_picos, incertidumbres, identificaciones)):
    print(f"Resonancia {i+1}:")
    print(f"  Masa = {masa:.3f} ± {incert:.3f} GeV")
    print(f"  Identificación: {ident}")
    print(f"  Incertidumbre relativa: {incert/masa*100:.2f}%")

