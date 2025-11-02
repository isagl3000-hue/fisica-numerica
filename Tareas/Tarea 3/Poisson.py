#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 21:40:36 2025

@author: isaias-gl
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def resolver_poisson_gauss_seidel(N=100, tol=1e-8, max_iter=10000):
    """
    Resuelve la ecuación de Poisson con condiciones periódicas usando Gauss-Seidel
    
    Parámetros:
    N: número de puntos en cada dirección
    tol: tolerancia para convergencia
    max_iter: máximo número de iteraciones
    """
    
    # Dominio [0, 2pi] x [0, 2pi]
    L = 2 * np.pi
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # Crear mallas
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Término fuente f(x,y)
    F = np.cos(3*X + 4*Y) - np.cos(5*X - 2*Y)
    
    # Solución inicial (cero)
    phi = np.zeros((N, N))
    
    # Factor para la ecuación discretizada
    factor = 1.0 / (2/dx**2 + 2/dy**2)
    
    print("Resolviendo ecuación de Poisson con Gauss-Seidel...")
    print(f"Grid: {N}x{N}, Tolerancia: {tol}")
    
    # Iteración de Gauss-Seidel
    for iteracion in range(max_iter):
        phi_old = phi.copy()
        max_error = 0.0
        
        # Actualizar solución punto por punto
        for i in range(N):
            for j in range(N):
                # Índices con condiciones periódicas
                ip1 = (i + 1) % N
                im1 = (i - 1) % N
                jp1 = (j + 1) % N
                jm1 = (j - 1) % N
                
                # Actualización Gauss-Seidel
                phi[i, j] = factor * (
                    (phi[ip1, j] + phi[im1, j]) / dx**2 +
                    (phi[i, jp1] + phi[i, jm1]) / dy**2 -
                    F[i, j]
                )
        
        # Calcular error máximo
        max_error = np.max(np.abs(phi - phi_old))
        
        if iteracion % 1000 == 0:
            print(f"Iteración {iteracion}: Error máximo = {max_error:.2e}")
        
        if max_error < tol:
            print(f"Convergencia alcanzada en {iteracion} iteraciones")
            print(f"Error final: {max_error:.2e}")
            break
    
    if iteracion == max_iter - 1:
        print(f"Advertencia: Máximo de iteraciones alcanzado")
        print(f"Error final: {max_error:.2e}")
    
    return X, Y, phi, F

def visualizar_solucion(X, Y, phi, F):
    """
    Visualiza la solución de la ecuación de Poisson
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Gráfica 1: Solución phi(x,y)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, phi, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('$\phi(x,y)$')
    ax1.set_title('Solución: $\phi(x,y)$')
    plt.colorbar(surf1, ax=ax1, shrink=0.6, label='$\phi$')
    
    # Gráfica 2: Fuente f(x,y)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, F, cmap='plasma', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('$f(x,y)$')
    ax2.set_title('Término fuente: $f(x,y)$')
    plt.colorbar(surf2, ax=ax2, shrink=0.6, label='$f$')
    
    # Gráfica 3: Contornos
    ax3 = fig.add_subplot(1, 3, 3)
    contour = ax3.contourf(X, Y, phi, levels=20, cmap='viridis')
    ax3.contour(X, Y, phi, levels=10, colors='black', linewidths=0.5)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Contornos de $\phi(x,y)$')
    ax3.set_aspect('equal')
    plt.colorbar(contour, ax=ax3, label='$\phi$')
    
    plt.tight_layout()
    plt.show()

def verificar_solucion(X, Y, phi, F, dx):
    """
    Verifica que la solución satisfaga la ecuación de Poisson
    """
    N = phi.shape[0]
    
    # Calcular Laplaciano numérico de la solución
    laplaciano = np.zeros_like(phi)
    
    for i in range(N):
        for j in range(N):
            ip1 = (i + 1) % N
            im1 = (i - 1) % N
            jp1 = (j + 1) % N
            jm1 = (j - 1) % N
            
            d2phidx2 = (phi[ip1, j] - 2*phi[i, j] + phi[im1, j]) / dx**2
            d2phidy2 = (phi[i, jp1] - 2*phi[i, j] + phi[i, jm1]) / dx**2
            
            laplaciano[i, j] = d2phidx2 + d2phidy2
    
    # Calcular residual
    residual = laplaciano - F
    error_max = np.max(np.abs(residual))
    error_rms = np.sqrt(np.mean(residual**2))
    
    print("\n=== VERIFICACIÓN DE LA SOLUCIÓN ===")
    print(f"Error máximo |∇²φ - f|: {error_max:.2e}")
    print(f"Error RMS |∇²φ - f|: {error_rms:.2e}")
    
    return residual

# Resolver la ecuación de Poisson
print("=== ECUACIÓN DE POISSON CON CONDICIONES PERIÓDICAS ===")
X, Y, phi, F = resolver_poisson_gauss_seidel(N=100, tol=1e-8)

# Visualizar resultados
visualizar_solucion(X, Y, phi, F)

# Verificar la solución
residual = verificar_solucion(X, Y, phi, F, dx=2*np.pi/100)

# Gráfica adicional del residual
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.contourf(X, Y, residual, levels=20, cmap='RdBu_r')
plt.colorbar(label='Residual')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Residual: $\\nabla^2\\phi - f$')
plt.gca().set_aspect('equal')

plt.subplot(1, 2, 2)
plt.hist(residual.flatten(), bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Residual')
plt.ylabel('Frecuencia')
plt.title('Distribución del Residual')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()