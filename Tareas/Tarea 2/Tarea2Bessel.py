import numpy as np
import matplotlib.pyplot as plt

def bessel_up(x, l_max):
    """
    Calcula j_l(x) usando recurrencia hacia arriba
    """
    j = np.zeros(l_max + 1) #Creamos un arreglo inicializado en cero
    
    # Valores iniciales
    j[0] = np.sin(x) / x if x != 0 else 1.0
    if l_max >= 1:
        j[1] = np.sin(x) / (x**2) - np.cos(x) / x if x != 0 else 0.0
    
    # Recurrencia hacia arriba
    for l in range(1, l_max):
        j[l+1] = ((2*l + 1) / x) * j[l] - j[l-1]
    
    return j

def bessel_down(x, l_max):
    """
    Calcula j_l(x) usando recurrencia hacia abajo
    """
    # Empezamos desde l alto con valores semilla
    L_start = l_max + 50  # Empezamos más arriba para estabilidad
    j_high = np.zeros(L_start + 2)
    
    # Valores semilla para l alto (aproximación)
    j_high[L_start + 1] = 0.0
    j_high[L_start] = 1.0  # Valor arbitrario (se normalizará después)
    
    # Recurrencia hacia abajo
    for l in range(L_start, 0, -1):
        j_high[l-1] = ((2*l + 1) / x) * j_high[l] - j_high[l+1]
    
    # Normalización usando j_0 conocida
    j_0_exact = np.sin(x) / x if x != 0 else 1.0
    scale = j_0_exact / j_high[0]
    
    return j_high[:l_max + 1] * scale

# Parámetros
l_max = 24
x_values = [0.1, 1.0, 10.0]

print("CÁLCULO DE FUNCIONES DE BESSEL ESFÉRICAS")
print("=" * 60)

for x in x_values:
    print(f"\nPara x = {x}:")
    print("-" * 40)
    
    j_up = bessel_up(x, l_max)
    j_down = bessel_down(x, l_max)
    
    print(f"{'l':<3} {'j_up':<15} {'j_down':<15} {'Diferencia':<15}")
    print("-" * 60)
    
    for l in range(min(10, l_max + 1)):  # Mostrar solo primeros 10 valores
        diff = abs(j_up[l] - j_down[l])
        print(f"{l:<3} {j_up[l]:<15.6e} {j_down[l]:<15.6e} {diff:<15.6e}")
    
    if l_max > 10:
        print("...")
        # Mostrar algunos valores altos de l
        for l in [15, 20, 24]:
            diff = abs(j_up[l] - j_down[l])
            print(f"{l:<3} {j_up[l]:<15.6e} {j_down[l]:<15.6e} {diff:<15.6e}")


#GRAFICA (no necesaria para la tarea) ========================================
# Visualización para x = 1.0
x = 1.0
j_up = bessel_up(x, l_max)
j_down = bessel_down(x, l_max)
l_values = np.arange(l_max + 1)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.semilogy(l_values, np.abs(j_up), 'bo-', label='Hacia arriba', markersize=4)
plt.semilogy(l_values, np.abs(j_down), 'ro-', label='Hacia abajo', markersize=4)
plt.xlabel('l')
plt.ylabel('|j_l(x)|')
plt.title(f'Funciones de Bessel esféricas (x = {x})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
difference = np.abs(j_up - j_down)
plt.semilogy(l_values, difference, 'g-', linewidth=2)
plt.xlabel('l')
plt.ylabel('|j_up - j_down|')
plt.title('Diferencia entre métodos')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
# Comparación para x = 0.1
x_small = 0.1
j_up_small = bessel_up(x_small, l_max)
j_down_small = bessel_down(x_small, l_max)
plt.semilogy(l_values, np.abs(j_up_small), 'bo-', markersize=4)
plt.semilogy(l_values, np.abs(j_down_small), 'ro-', markersize=4)
plt.xlabel('l')
plt.ylabel('|j_l(x)|')
plt.title(f'Funciones de Bessel (x = {x_small})')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
# Comparación para x = 10.0
x_large = 10.0
j_up_large = bessel_up(x_large, l_max)
j_down_large = bessel_down(x_large, l_max)
plt.semilogy(l_values, np.abs(j_up_large), 'bo-', markersize=4)
plt.semilogy(l_values, np.abs(j_down_large), 'ro-', markersize=4)
plt.xlabel('l')
plt.ylabel('|j_l(x)|')
plt.title(f'Funciones de Bessel (x = {x_large})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
#=============================================================================


# VALORES DE REFERENCIA DE LA TABLA
reference_values = {
    0.1: {
        3: 9.518519719e-6,
        5: 9.616310231e-10,
        8: 2.901200102e-16
    },
    1.0: {
        3: 9.006581118e-3,
        5: 9.256115862e-5,
        8: 2.826498802e-8
    },
    10.0: {
        3: -3.949584498e-2,
        5: -5.553451162e-2,
        8: 1.255780236e-1
    }
}

def verificacion():
    """
    Verifica la precisión de los métodos comparando con valores de referencia
    """
    l_max = 24
    x_values = [0.1, 1.0, 10.0]
    l_prueba = [3, 5, 8]
    
    print("VERIFICACIÓN DE PRECISIÓN - INCISO (b)")
    print("=" * 70)
    print("Comparación con valores de referencia de la tabla")
    print("Error relativo = |calculado - referencia| / |referencia|")
    print("=" * 70)
    
    for x in x_values:
        print(f"\n>>> PARA x = {x}:")
        print("-" * 70)
        
        # Calcular con ambos métodos
        j_up = bessel_up(x, l_max)
        j_down = bessel_down(x, l_max)
        
        print(f"{'l':<3} {'Referencia':<15} {'Hacia arriba':<15} {'Hacia abajo':<15} {'Error↑':<12} {'Error↓':<12}")
        print("-" * 70)
        
        for l in l_prueba:
            ref_val = reference_values[x][l]
            up_val = j_up[l]
            down_val = j_down[l]
            
            error_up = abs(up_val - ref_val) / abs(ref_val) if ref_val != 0 else abs(up_val - ref_val)
            error_down = abs(down_val - ref_val) / abs(ref_val) if ref_val != 0 else abs(down_val - ref_val)
            
            # Marcar si cumple el criterio de 10^-10
            up_ok = "✓" if error_up < 1e-10 else "✗"
            down_ok = "✓" if error_down < 1e-10 else "✗"
            
            print(f"{l:<3} {ref_val:<15.6e} {up_val:<15.6e} {down_val:<15.6e} "
                  f"{error_up:.2e}{up_ok} {error_down:.2e}{down_ok}")
    
 
# Ejecutar la verificación
verificacion()

# Cálculo completo para los primeros 25 valores de l
print("\n" + "=" * 70)
print("CÁLCULO COMPLETO PARA LOS PRIMEROS 25 VALORES DE l")
print("=" * 70)

for x in [0.1, 1.0, 10.0]:
    print(f"\n--- x = {x} ---")
    j_down = bessel_down(x, 24)  # Usamos el método confiable
    
    print(f"{'l':<3} {'j_l(x)':<20}")
    print("-" * 30)
    for l in range(25):
        print(f"{l:<3} {j_down[l]:<20.10e}")
        
        
def comparacion():
    """
    Inciso (c): Comparar métodos con distintas fórmulas de recurrencia
    """
    l_max = 24
    x_values = [0.1, 1.0, 10.0]
    
    print("INCISO (c): COMPARACIÓN DE MÉTODOS DE RECURRENCIA")
    print("=" * 90)
    print("Tabla: | l | j_l^{up} | j_l^{down} | |j_l^{up} - j_l^{down}| / (|j_l^{up}| + |j_l^{down}|) |")
    print("=" * 90)
    
    for x in x_values:
        print(f"\n>>> PARA x = {x}")
        print("-" * 90)
        
        # Calcular con ambos métodos
        j_up = bessel_up(x, l_max)
        j_down = bessel_down(x, l_max)
        
        # Imprimir tabla
        print(f"{'l':<3} {'j_up':<15} {'j_down':<15} {'Diferencia Relativa':<20}")
        print("-" * 90)
        
        for l in range(l_max + 1):
            j_up_val = j_up[l]
            j_down_val = j_down[l]
            
            # Calcular diferencia relativa según la fórmula del problema
            numerator = abs(j_up_val - j_down_val)
            denominator = abs(j_up_val) + abs(j_down_val)
            
            if denominator > 0:
                rel_diff = numerator / denominator
            else:
                rel_diff = 0.0
            
            # Formatear salida
            print(f"{l:<3} {j_up_val:<15.6e} {j_down_val:<15.6e} {rel_diff:<20.6e}")
            
           
    
    return j_up, j_down

# Ejecutar comparación
j_up, j_down = comparacion()












