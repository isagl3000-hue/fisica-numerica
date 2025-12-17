#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 11:10:36 2025

@author: isaias-gl
"""
import numpy as np
import scipy.stats as stats
from scipy import integrate

def integracion_monte_carlo(funcion, limite_inferior, limite_superior, 
                           num_muestras=10000, confianza=0.95):
    """
    Estima la integral usando el método de Montecarlo
    y calcula el intervalo de confianza con fórmulas de recurrencia.
    
    Parámetros:
    funcion: función a integrar
    limite_inferior: límite inferior de integración
    limite_superior: límite superior de integración
    num_muestras: número de muestras aleatorias
    confianza: nivel de confianza (ej. 0.95 para 95%)
    
    Retorna:
    estimacion: estimación de la integral
    intervalo_inf: límite inferior del intervalo de confianza
    intervalo_sup: límite superior del intervalo de confianza
    """
    # Generar puntos aleatorios uniformemente distribuidos
    puntos_aleatorios = np.random.uniform(limite_inferior, limite_superior, num_muestras)
    
    # Evaluar la función en los puntos aleatorios
    valores_funcion = funcion(puntos_aleatorios)
    
    # Estimación de Montecarlo: (b-a) * promedio de f(x)
    estimacion = (limite_superior - limite_inferior) * np.mean(valores_funcion)
    
    # Calcular varianza y desviación estándar de los valores de la función
    varianza_funcion = np.var(valores_funcion, ddof=1)  # ddof=1 para varianza muestral
    
    # Calcular varianza de la estimación
    varianza_estimacion = ((limite_superior - limite_inferior)**2 * 
                          varianza_funcion) / num_muestras
    
    # Calcular error estándar (desviación estándar de la estimación)
    error_estandar = np.sqrt(varianza_estimacion)
    
    # Calcular intervalo de confianza usando la distribución t para mayor precisión
    # con muestras pequeñas, o normal para muestras grandes
    if num_muestras < 30:
        # Usar distribución t para muestras pequeñas
        grados_libertad = num_muestras - 1
        valor_t = stats.t.ppf((1 + confianza) / 2, grados_libertad)
        intervalo_inf = estimacion - valor_t * error_estandar
        intervalo_sup = estimacion + valor_t * error_estandar
    else:
        # Usar distribución normal para muestras grandes
        valor_z = stats.norm.ppf((1 + confianza) / 2)
        intervalo_inf = estimacion - valor_z * error_estandar
        intervalo_sup = estimacion + valor_z * error_estandar
    
    return estimacion, intervalo_inf, intervalo_sup

def ejercicio_1a(num_muestras):
    
    print("=" * 60)
    print("Ejercicio 1(a):")
    print("=" * 60)
    
    # Definir la función
    def funcion_1a(x):
        return (1 - x**2)**(3/2)
    
    # Límites de integración
    limite_inf, limite_sup = -1, 1
    
    # Calcular con Montecarlo
    estimacion, intervalo_inf, intervalo_sup = integracion_monte_carlo(
        funcion_1a, limite_inf, limite_sup, num_muestras)
    
    
    valor_exacto, error = integrate.quad(funcion_1a,limite_inf,limite_sup)
    
    print(f"Número de muestras: {num_muestras}")
    print(f"Estimación Montecarlo: {estimacion:.8f}")
    print(f"Valor exacto (analítico): {valor_exacto:.8f}")
    print(f"Intervalo de confianza al 95%: [{intervalo_inf:.8f}, {intervalo_sup:.8f}]")
    print(f"Error absoluto: {abs(estimacion - valor_exacto):.8f}")
    print(f"Error relativo: {abs(estimacion - valor_exacto)/abs(valor_exacto)*100:.6f}%")
    print()

def ejercicio_1b(num_muestras):
    
    print("=" * 60)
    print("Ejercicio 1(b):")
    print("=" * 60)
    
    # Definir la función
    def funcion_1b(x):
        return np.exp(x + x**2)
    
    # Límites de integración
    limite_inf, limite_sup = -4, 4
    
    # Calcular con Montecarlo
    estimacion, intervalo_inf, intervalo_sup = integracion_monte_carlo(
        funcion_1b, limite_inf, limite_sup, num_muestras)
    
    # Calcular valor de referencia usando scipy.integrate.quad
    valor_referencia, error_referencia = integrate.quad(funcion_1b, limite_inf, limite_sup)
    
    print(f"Número de muestras: {num_muestras}")
    print(f"Estimación Montecarlo: {estimacion:.8f}")
    print(f"Valor de referencia: {valor_referencia:.8f}")
    print(f"Intervalo de confianza al 95%: [{intervalo_inf:.8f}, {intervalo_sup:.8f}]")
    print(f"Error absoluto: {abs(estimacion - valor_referencia):.8f}")
    print(f"Error relativo: {abs(estimacion - valor_referencia)/abs(valor_referencia)*100:.6f}%")
    print()

def ejecutar_estudio_convergencia():
    """Ejecuta el estudio de convergencia con diferentes números de muestras"""
    print("FÍSICA NUMÉRICA - ACTIVIDAD #12")
    print("MÉTODOS DE MONTE CARLO - ESTUDIO DE CONVERGENCIA")
    print("=" * 60)
    
    # Establecer semilla para reproducibilidad
    np.random.seed(42)
    
    # Valores de número de muestras para el estudio
    valores_muestras = [1000, 10000, 100000]
    
    for num_muestras in valores_muestras:
        print(f"\n{'='*60}")
        print(f"ESTUDIO CON {num_muestras} MUESTRAS")
        print(f"{'='*60}")
        
        # Ejecutar ejercicio 1(a)
        ejercicio_1a(num_muestras)
        
        # Ejecutar ejercicio 1(b)
        ejercicio_1b(num_muestras)
    
    

def main():
    """Función principal del programa"""
    ejecutar_estudio_convergencia()
    
    

if __name__ == "__main__":
    main()