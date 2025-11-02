#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 00:03:48 2025

@author: isaias-gl
"""
import math


#PUNTO 1 ----------------------------------------------------------------------
def overflow():
    x = 1.0
    while True:
        try:
            x_prev = x
            x *= 2.0
            if x == float('inf'):
                return x_prev
        except:
            return x_prev

def underflow():
    x = 1.0
    while True:
        x_prev = x
        x /= 2.0
        if x == 0.0:
            return x_prev

overflow_limit = overflow()
underflow_limit = underflow()

print(f"Límite de overflow (dentro de un factor de 2): {overflow_limit:.3e}")
print(f"Límite de underflow (dentro de un factor de 2): {underflow_limit:.3e}")


#PUNTO 2 ----------------------------------------------------------------------
def precision_maquina():
    epsilon = 1.0
    while 1.0 + epsilon != 1.0:
        epsilon_prev = epsilon
        epsilon /= 2.0
    
    return epsilon_prev

# Calcular la precisión de máquina
epsilon_m = precision_maquina()

print(f"Precisión de máquina εₘ: {epsilon_m:.20e}")
print(f"Valor teórico esperado:  2.22e-16")

# Verificación
print("\nVerificación:")
print(f"1.0 + εₘ = {1.0 + epsilon_m:.20f}")
print(f"¿1.0 + εₘ/2 == 1.0? {1.0 + epsilon_m/2 == 1.0}")


#PUNTO 3 ----------------------------------------------------------------------

def seno_taylor(x, tolerancia=epsilon_m/2): #1e-8
    """
    Calcula sen(x) usando series de Taylor con error absoluto < tolerancia
    """
    # Asegurar que x esté en el rango [-2π, 2π] usando periodicidad
    #x = x % (2 * math.pi)
    
    termino = x  # Primer término: x
    suma = termino
    n = 1
    error_relativo = 1.0
    
    # Almacenar resultados para la tabla
    resultados = []
    
    while abs(termino) > tolerancia:
        # Calcular siguiente término usando relación de recurrencia
        # término_{n+1} = -término_n * x² / [(2n)(2n+1)]
        termino = -termino * x * x / ((2*n) * (2*n + 1))
        suma += termino
        n += 1
        
        # Calcular error relativo
        valor_real = math.sin(x)
        error_relativo = abs((suma - valor_real) / valor_real) if valor_real != 0 else abs(suma)
        
        resultados.append([n, suma, error_relativo])
        
        # Prevenir bucles infinitos
        if n > 1000:
            break
    
    return suma, resultados


print("="*70)
print("PUNTO 3(a): CÁLCULO DE sen(x) POR SERIES DE TAYLOR")
print("="*70)

x=0.5
aproximacion, tabla = seno_taylor(x)
real = math.sin(x)
error_absoluto = abs(aproximacion - real)

print(f"\nx = {x:.2f}")
print(f"Valor real: {real:.10f}")
print(f"Aproximación: {aproximacion:.10f}")
print(f"Error absoluto: {error_absoluto:.2e}")
print(f"Error relativo: {abs((aproximacion - real)/real):.2e}")

# Mostrar primeras filas de la tabla
print("\nTabla (primeras 5 iteraciones):")
print("N    Suma                    Error Relativo")
print("-" * 50)
for i in range(min(5, len(tabla))):
    n, suma, error = tabla[i]
    print(f"{n:2d}  {suma:10.8f}  {error:12.2e}")



def calcular_seno_comparacion(x, tolerancia=epsilon_m/2): #1e-8
    """
    Compara el cálculo de sen(x) con y sin usar periodicidad
    """
    x_reducido = x % (2 * math.pi)
    
    # Calcular con periodicidad (método correcto)
    aprox_periodico = seno_taylor(x_reducido, tolerancia)[0] #[0] especifica que solo tomaremos el 1er miembro de la tupla de la funcion seno_taylor
    
    # Calcular sin periodicidad (método problemático)
    aprox_directo = seno_taylor(x, tolerancia)[0]
    
    real = math.sin(x)
    
    return x_reducido, aprox_periodico, aprox_directo, real



# Tabla comparativa
print("="*90)
print("PUNTO 3(b): COMPARACIÓN CON Y SIN PERIODICIDAD")
print("="*90)
print(f"{'x':>8} {'x_reducido':>12} {'Real':>15} {'Con periodicidad':>18} {'Sin periodicidad':>18} {'Error Periodic':>15} {'Error Directo':>15}")
print("-"*90)

valores_test = [10, 20, 30, 40, 50, 100, 200]

for x in valores_test:
    x_red, aprox_per, aprox_dir, real = calcular_seno_comparacion(x)
    
    error_per = abs(aprox_per - real)
    error_dir = abs(aprox_dir - real)
    
    print(f"{x:8.1f} {x_red:12.6f} {real:15.10f} {aprox_per:18.10f} {aprox_dir:18.10f} {error_per:15.2e} {error_dir:15.2e}")

print("="*90)












