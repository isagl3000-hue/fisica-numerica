#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 23:10:06 2025

@author: isaias-gl
"""
import numpy as np
import matplotlib.pyplot as plt

# ===================================================
# CONSTANTES FÍSICAS
# ===================================================
masa_pion = 139.6  # MeV/c^2 (masa del pión)
vida_media_reposo = 2.6e-8  # s (vida media en reposo)
velocidad_luz = 3.0e8  # m/s (velocidad de la luz)
distancia = 20.0  # m (distancia a recorrer)

# ===================================================
# FUNCIONES AUXILIARES
# ===================================================
def calcular_gamma(energia_cinetica, masa):
    """Calcula el factor gamma dada la energía cinética T."""
    return 1.0 + energia_cinetica/masa

def calcular_velocidad(gamma):
    """Calcula la velocidad (fracción de c) dado gamma."""
    return np.sqrt(1.0 - 1.0/(gamma**2))

def calcular_probabilidad_supervivencia(energia_cinetica, masa, vida_media, distancia):
    """
    Calcula la probabilidad de que un pión sobreviva
    después de recorrer 'distancia' metros.
    """
    gamma = calcular_gamma(energia_cinetica, masa)
    velocidad = calcular_velocidad(gamma) * velocidad_luz  # m/s
    vida_media_lab = gamma * vida_media  # s (vida media en laboratorio)
    # Probabilidad de supervivencia: exp(-t/vida_media_lab)
    tiempo_viaje = distancia / velocidad  # tiempo de viaje en laboratorio
    return np.exp(-tiempo_viaje / vida_media_lab)

# ===================================================
# PARTE (a): PIONES MONOENERGÉTICOS
# ===================================================
def piones_monoenergeticos():
    """Simulación con energía fija de 200 MeV."""
    energia_fija = 200.0  # MeV
    total_piones = 1_000_000
    
    # Probabilidad de supervivencia individual
    prob_supervivencia = calcular_probabilidad_supervivencia(
        energia_fija, masa_pion, vida_media_reposo, distancia
    )
    
    # Simulación binomial: cada pión sobrevive con probabilidad prob_supervivencia
    piones_sobrevivientes = np.random.binomial(total_piones, prob_supervivencia)
    
    # Resultados teóricos para comparación
    gamma = calcular_gamma(energia_fija, masa_pion)
    velocidad = calcular_velocidad(gamma) * velocidad_luz
    vida_media_laboratorio = gamma * vida_media_reposo
    tiempo_recorrido = distancia / velocidad
    
    print("="*60)
    print("PARTE (a): PIONES MONOENERGÉTICOS (200 MeV)")
    print("="*60)
    print(f"Número total de piones: {total_piones:,}")
    print(f"Factor gamma (γ): {gamma:.4f}")
    print(f"Velocidad: {velocidad/velocidad_luz:.4f} c = {velocidad:.2e} m/s")
    print(f"Tiempo de vida en laboratorio: {vida_media_laboratorio:.2e} s")
    print(f"Tiempo para recorrer {distancia} m: {tiempo_recorrido:.2e} s")
    print(f"Probabilidad individual de supervivencia: {prob_supervivencia:.6f}")
    print(f"Piones que sobreviven (simulación): {piones_sobrevivientes:,}")
    print(f"Piones que sobreviven (teórico esperado): {int(total_piones * prob_supervivencia):,}")
    print(f"Fracción que sobrevive: {piones_sobrevivientes/total_piones:.6f}")
    
    return piones_sobrevivientes, prob_supervivencia

# ===================================================
# PARTE (b): PIONES CON DISTRIBUCIÓN GAUSSIANA
# ===================================================
def piones_energia_gaussiana():
    """Simulación con energía distribuida normalmente."""
    total_piones = 1_000_000
    energia_media = 200.0  # MeV
    desviacion_energia = 50.0   # MeV
    
    # Generar energías con distribución normal
    # (truncamos a energías positivas por física)
    energias = np.random.normal(energia_media, desviacion_energia, total_piones)
    energias = np.maximum(energias, 1.0)  # Energía mínima 1 MeV
    
    # Calcular probabilidades de supervivencia para cada pión
    probabilidades_supervivencia = np.zeros(total_piones)
    for i in range(total_piones):
        probabilidades_supervivencia[i] = calcular_probabilidad_supervivencia(
            energias[i], masa_pion, vida_media_reposo, distancia
        )
    
    # Simular supervivencia de cada pión
    numeros_aleatorios = np.random.random(total_piones)
    mascara_sobrevivientes = numeros_aleatorios < probabilidades_supervivencia
    piones_sobrevivientes = np.sum(mascara_sobrevivientes)
    
    # Estadísticas
    probabilidad_media_supervivencia = np.mean(probabilidades_supervivencia)
    
    print("\n" + "="*60)
    print("PARTE (b): PIONES CON ENERGÍA GAUSSIANA")
    print("="*60)
    print(f"Energía media: {energia_media} MeV")
    print(f"Desviación estándar: {desviacion_energia} MeV")
    print(f"Energía mínima generada: {energias.min():.1f} MeV")
    print(f"Energía máxima generada: {energias.max():.1f} MeV")
    print(f"Energía media real: {energias.mean():.1f} MeV")
    print(f"Probabilidad media de supervivencia: {probabilidad_media_supervivencia:.6f}")
    print(f"Piones que sobreviven (simulación): {piones_sobrevivientes:,}")
    print(f"Fracción que sobrevive: {piones_sobrevivientes/total_piones:.6f}")
    
    return energias, probabilidades_supervivencia, piones_sobrevivientes


# ===================================================
# EJECUCIÓN PRINCIPAL
# ===================================================
if __name__ == "__main__":
    print("SIMULACIÓN DE DECAIMIENTO DE PIONES")
    print("="*60)
    
    # Parte (a)
    print("\nEjecutando simulación monoenergética...")
    sobrevivientes_mono, prob_supervivencia_mono = piones_monoenergeticos()
    
    # Parte (b)
    print("\nEjecutando simulación con distribución gaussiana...")
    energias_gauss, probs_supervivencia_gauss, sobrevivientes_gauss = piones_energia_gaussiana()
    
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN COMPARATIVO")
    print("="*60)
    print(f"Monoenergético (200 MeV): {sobrevivientes_mono:,} sobrevivientes")
    print(f"Gaussiano (200±50 MeV):   {sobrevivientes_gauss:,} sobrevivientes")
    diferencia = abs(sobrevivientes_mono - sobrevivientes_gauss)
    print(f"Diferencia: {diferencia:,} piones")
    print(f"Diferencia relativa: {diferencia/sobrevivientes_mono*100:.2f}%")
    
    # Información adicional
    print("\n" + "="*60)
    print("INFORMACIÓN ADICIONAL")
    print("="*60)
    print("Nota: Los resultados pueden variar ligeramente entre ejecuciones")
    print("debido a la naturaleza aleatoria de la simulación.")
    print("\nInterpretación física:")
    print("- Los piones con menor energía tienen menor factor gamma")
    print("- Menor gamma = menor tiempo de vida en el laboratorio")
    print("- Por lo tanto, mayor probabilidad de decaer antes de los 20 metros")