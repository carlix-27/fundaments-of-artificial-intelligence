"""
Práctica Bloque 5: Minimax para Tateti
======================================

Curso: Fundamentos e Historia de la Inteligencia Artificial
Universidad Austral

Instrucciones:
1. Completa las funciones marcadas con # TODO
2. Ejecuta el archivo para verificar tu implementación
3. Compara tus resultados con los valores esperados

Valores esperados:
- Valor del juego: 0 (empate)
- Mejor movimiento: (1, 1) - centro
- Nodos explorados (Minimax): ~549,946
"""

import time
from typing import List, Tuple, Optional

# ==============================================================================
# REPRESENTACIÓN DEL ESTADO
# ==============================================================================

# El tablero es una lista de 3 listas, cada una con 3 elementos
# ' ' = vacío, 'X' = MAX, 'O' = MIN
EstadoTateti = List[List[str]]

def crear_tablero_vacio() -> EstadoTateti:
    """Crea y retorna un tablero vacío 3x3."""
    return [[' ', ' ', ' '],
            [' ', ' ', ' '],
            [' ', ' ', ' ']]


def imprimir_tablero(estado: EstadoTateti) -> None:
    """Imprime el tablero de forma legible."""
    print()
    for i, fila in enumerate(estado):
        print(f" {fila[0]} | {fila[1]} | {fila[2]} ")
        if i < 2:
            print("---+---+---")
    print()


# ==============================================================================
# FUNCIONES A IMPLEMENTAR
# ==============================================================================

def acciones(estado: EstadoTateti) -> List[Tuple[int, int]]:
    """
    Retorna lista de acciones válidas (casillas vacías).
    
    Parámetros:
        estado: tablero actual
    
    Retorna:
        Lista de tuplas (fila, columna) donde hay casillas vacías
    
    Ejemplo:
        estado = [['X', ' ', ' '], [' ', 'O', ' '], [' ', ' ', ' ']]
        acciones(estado) → [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2)]
    """
    acciones_validas: List[Tuple[int, int]] = []

    for fila in range(3):
        for columna in range(3):
            if estado[fila][columna] == ' ':
                acciones_validas.append((fila, columna))

    return acciones_validas


def acciones_ordenadas(estado: EstadoTateti) -> List[Tuple[int, int]]:
    prioridad = {
        (1, 1): 0,
        (0, 0): 1,
        (0, 2): 1,
        (2, 0): 1,
        (2, 2): 1,
        (0, 1): 2,
        (1, 0): 2,
        (1, 2): 2,
        (2, 1): 2,
    }

    return sorted(acciones(estado), key=lambda accion: prioridad.get(accion, 3))


def resultado(estado: EstadoTateti, accion: Tuple[int, int], jugador: str) -> EstadoTateti:
    """
    Aplica la acción al estado y retorna el nuevo estado.
    
    ¡IMPORTANTE! No modificar el estado original, crear una COPIA.
    
    Parámetros:
        estado: tablero actual
        accion: tupla (fila, columna) donde jugar
        jugador: 'X' o 'O'
    
    Retorna:
        Nuevo estado con la jugada aplicada
    
    Ejemplo:
        estado = [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']]
        resultado(estado, (1, 1), 'X') → [[' ', ' ', ' '], [' ', 'X', ' '], [' ', ' ', ' ']]
    """
    nuevo_estado = [fila.copy() for fila in estado]
    fila, columna = accion
    nuevo_estado[fila][columna] = jugador
    return nuevo_estado


def verificar_ganador(estado: EstadoTateti) -> Optional[str]:
    """
    Verifica si hay un ganador.
    
    Retorna:
        'X' si ganó X
        'O' si ganó O
        None si no hay ganador
    """
    # Todas las líneas posibles (8 en total)
    lineas = [
        # Filas
        [(0, 0), (0, 1), (0, 2)],
        [(1, 0), (1, 1), (1, 2)],
        [(2, 0), (2, 1), (2, 2)],
        # Columnas
        [(0, 0), (1, 0), (2, 0)],
        [(0, 1), (1, 1), (2, 1)],
        [(0, 2), (1, 2), (2, 2)],
        # Diagonales
        [(0, 0), (1, 1), (2, 2)],
        [(0, 2), (1, 1), (2, 0)]
    ]
    
    for linea in lineas:
        valores = [estado[fila][columna] for fila, columna in linea]
        if valores[0] != ' ' and valores[0] == valores[1] == valores[2]:
            return valores[0]

    return None


def es_terminal(estado: EstadoTateti) -> bool:
    """
    Verifica si el estado es terminal (victoria o empate).
    
    Retorna:
        True si el juego terminó
        False si el juego continúa
    """
    return verificar_ganador(estado) is not None or not acciones(estado)


def utilidad(estado: EstadoTateti) -> int:
    """
    Retorna la utilidad del estado terminal.
    
    Solo llamar si es_terminal(estado) == True
    
    Retorna:
        +1 si ganó X (MAX)
        -1 si ganó O (MIN)
         0 si empate
    """
    ganador = verificar_ganador(estado)

    if ganador == 'X':
        return 1
    if ganador == 'O':
        return -1
    return 0


# ==============================================================================
# ALGORITMO MINIMAX
# ==============================================================================

# Contador global de nodos explorados
nodos_explorados = 0


def minimax(estado: EstadoTateti, es_turno_max: bool) -> int:
    """
    Algoritmo Minimax recursivo.
    
    Parámetros:
        estado: estado actual del juego
        es_turno_max: True si es turno de MAX (X), False si es MIN (O)
    
    Retorna:
        El valor minimax del estado (-1, 0, o +1)
    """
    global nodos_explorados
    nodos_explorados += 1
    
    if es_terminal(estado):
        return utilidad(estado)

    if es_turno_max:
        valor = -2
        for accion in acciones(estado):
            nuevo_estado = resultado(estado, accion, 'X')
            valor = max(valor, minimax(nuevo_estado, False))
        return valor

    valor = 2
    for accion in acciones(estado):
        nuevo_estado = resultado(estado, accion, 'O')
        valor = min(valor, minimax(nuevo_estado, True))
    return valor


def mejor_movimiento(estado: EstadoTateti) -> Tuple[int, int]:
    """
    Encuentra el mejor movimiento para X (MAX).
    
    Retorna:
        Tupla (fila, columna) del mejor movimiento
    """
    mejor_valor = -2
    mejor_accion: Optional[Tuple[int, int]] = None

    for accion in acciones_ordenadas(estado):
        valor = minimax(resultado(estado, accion, 'X'), False)
        if valor > mejor_valor:
            mejor_valor = valor
            mejor_accion = accion

    if mejor_accion is None:
        raise ValueError("No hay movimientos válidos disponibles")

    return mejor_accion


# ==============================================================================
# PODA ALFA-BETA
# ==============================================================================

nodos_alfabeta = 0


def minimax_alfabeta(estado: EstadoTateti, es_turno_max: bool, 
                     alfa: float, beta: float) -> int:
    """
    Minimax con poda Alfa-Beta.
    
    Parámetros:
        estado: estado actual
        es_turno_max: True si turno de MAX
        alfa: mejor valor que MAX puede garantizar (piso)
        beta: mejor valor que MIN puede garantizar (techo)
    
    Retorna:
        Valor minimax del estado
    """
    global nodos_alfabeta
    nodos_alfabeta += 1
    
    if es_terminal(estado):
        return utilidad(estado)

    if es_turno_max:
        valor = -2
        for accion in acciones(estado):
            valor = max(valor, minimax_alfabeta(resultado(estado, accion, 'X'), False, alfa, beta))
            alfa = max(alfa, valor)
            if beta <= alfa:
                break
        return valor

    valor = 2
    for accion in acciones(estado):
        valor = min(valor, minimax_alfabeta(resultado(estado, accion, 'O'), True, alfa, beta))
        beta = min(beta, valor)
        if beta <= alfa:
            break
    return valor


def mejor_movimiento_alfabeta(estado: EstadoTateti) -> Tuple[int, int]:
    """
    Encuentra el mejor movimiento usando Alfa-Beta.
    """
    mejor_valor = -2
    mejor_accion: Optional[Tuple[int, int]] = None
    alfa = float('-inf')
    beta = float('inf')

    for accion in acciones_ordenadas(estado):
        valor = minimax_alfabeta(resultado(estado, accion, 'X'), False, alfa, beta)
        if valor > mejor_valor:
            mejor_valor = valor
            mejor_accion = accion
        alfa = max(alfa, mejor_valor)

    if mejor_accion is None:
        raise ValueError("No hay movimientos válidos disponibles")

    return mejor_accion


# ==============================================================================
# VERIFICACIÓN Y PRUEBAS
# ==============================================================================

def verificar_implementacion():
    """Verifica que la implementación sea correcta."""
    print("Ejecutando tests de verificación...")
    print("-" * 40)
    
    # Test 1: acciones() en tablero vacío
    estado = crear_tablero_vacio()
    acc = acciones(estado)
    assert acc is not None, "acciones() retorna None"
    assert len(acc) == 9, f"Tablero vacío debería tener 9 acciones, tiene {len(acc)}"
    print("✓ Test 1: acciones() en tablero vacío")
    
    # Test 2: resultado() no modifica original
    estado = crear_tablero_vacio()
    nuevo = resultado(estado, (1, 1), 'X')
    assert estado[1][1] == ' ', "resultado() modificó el estado original!"
    assert nuevo[1][1] == 'X', "resultado() no aplicó la acción"
    print("✓ Test 2: resultado() crea copia correctamente")
    
    # Test 3: verificar_ganador()
    estado_x_gana = [['X', 'X', 'X'], ['O', 'O', ' '], [' ', ' ', ' ']]
    assert verificar_ganador(estado_x_gana) == 'X', "No detecta victoria de X en fila"
    
    estado_o_gana = [['X', 'O', 'X'], [' ', 'O', ' '], ['X', 'O', ' ']]
    assert verificar_ganador(estado_o_gana) == 'O', "No detecta victoria de O en columna"
    
    estado_diag = [['X', 'O', 'O'], [' ', 'X', ' '], [' ', ' ', 'X']]
    assert verificar_ganador(estado_diag) == 'X', "No detecta victoria en diagonal"
    print("✓ Test 3: verificar_ganador() funciona correctamente")
    
    # Test 4: es_terminal() y utilidad()
    assert es_terminal(estado_x_gana) == True
    assert utilidad(estado_x_gana) == 1
    
    estado_empate = [['X', 'O', 'X'], ['X', 'X', 'O'], ['O', 'X', 'O']]
    assert es_terminal(estado_empate) == True
    assert utilidad(estado_empate) == 0
    print("✓ Test 4: es_terminal() y utilidad() funcionan")
    
    # Test 5: Minimax desde estado inicial
    global nodos_explorados
    nodos_explorados = 0
    
    estado = crear_tablero_vacio()
    valor = minimax(estado, True)
    
    assert valor == 0, f"Valor desde inicio debería ser 0 (empate), obtuve {valor}"
    print(f"✓ Test 5: minimax() retorna 0 (exploró {nodos_explorados:,} nodos)")
    
    # Test 6: Mejor movimiento
    mov = mejor_movimiento(estado)
    assert mov == (1, 1), f"Mejor movimiento debería ser centro (1,1), obtuve {mov}"
    print("✓ Test 6: mejor_movimiento() retorna (1,1)")
    
    # Test 7: X puede ganar en 1
    estado = [['X', 'X', ' '], ['O', 'O', ' '], [' ', ' ', ' ']]
    nodos_explorados = 0
    valor = minimax(estado, True)
    mov = mejor_movimiento(estado)
    assert valor == 1, "X debería poder ganar"
    assert mov == (0, 2), "X debe jugar (0,2) para ganar"
    print("✓ Test 7: Detecta victoria forzada")
    
    print("-" * 40)
    print("¡Todos los tests pasaron! ✓")
    return True


def ejecutar_analisis():
    """Ejecuta el análisis completo y muestra resultados."""
    global nodos_explorados, nodos_alfabeta
    
    print("\n" + "=" * 50)
    print("ANÁLISIS DE TATETI CON MINIMAX")
    print("=" * 50)
    
    estado = crear_tablero_vacio()
    
    # Minimax puro
    print("\n1. MINIMAX PURO")
    print("-" * 30)
    nodos_explorados = 0
    
    inicio = time.time()
    valor = minimax(estado, True)
    mov = mejor_movimiento(estado)
    tiempo = time.time() - inicio
    
    print(f"   Valor del juego: {valor}")
    print(f"   Mejor movimiento: {mov}")
    print(f"   Nodos explorados: {nodos_explorados:,}")
    print(f"   Tiempo: {tiempo:.3f} segundos")
    
    # Alfa-Beta (si está implementado)
    if minimax_alfabeta(estado, True, float('-inf'), float('inf')) is not None:
        print("\n2. MINIMAX CON ALFA-BETA")
        print("-" * 30)
        nodos_alfabeta = 0
        
        inicio = time.time()
        valor_ab = minimax_alfabeta(estado, True, float('-inf'), float('inf'))
        tiempo_ab = time.time() - inicio
        
        print(f"   Valor del juego: {valor_ab}")
        print(f"   Nodos explorados: {nodos_alfabeta:,}")
        print(f"   Tiempo: {tiempo_ab:.3f} segundos")
        
        print("\n3. COMPARACIÓN")
        print("-" * 30)
        mejora = nodos_explorados / nodos_alfabeta if nodos_alfabeta > 0 else 0
        print(f"   Reducción de nodos: {mejora:.1f}x")
        print(f"   Nodos podados: {nodos_explorados - nodos_alfabeta:,} ({100*(1-nodos_alfabeta/nodos_explorados):.1f}%)")
    else:
        print("\n[Alfa-Beta no implementado]")
    
    print("\n" + "=" * 50)


def jugar_vs_ia():
    """Modo interactivo para jugar contra la IA."""
    print("\n¡Jugamos Tateti! Vos sos O, la IA es X.")
    print("Ingresá tu jugada como: fila,columna (ej: 1,1 para el centro)")
    print()
    
    estado = crear_tablero_vacio()
    
    while not es_terminal(estado):
        # Turno de X (IA)
        if len(acciones(estado)) % 2 == 1:  # X juega en turnos impares
            print("Turno de X (IA pensando...)")
            mov = mejor_movimiento(estado)
            estado = resultado(estado, mov, 'X')
            print(f"IA juega: {mov}")
        else:
            # Turno de O (humano)
            imprimir_tablero(estado)
            while True:
                try:
                    entrada = input("Tu turno (fila,col): ")
                    fila, col = map(int, entrada.split(','))
                    if (fila, col) in acciones(estado):
                        estado = resultado(estado, (fila, col), 'O')
                        break
                    else:
                        print("Casilla ocupada o inválida, intentá de nuevo.")
                except:
                    print("Formato inválido. Usá: fila,columna (ej: 0,2)")
    
    imprimir_tablero(estado)
    ganador = verificar_ganador(estado)
    if ganador == 'X':
        print("¡Ganó la IA! 🤖")
    elif ganador == 'O':
        print("¡Ganaste! 🎉")
    else:
        print("¡Empate! 🤝")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("PRÁCTICA BLOQUE 5: MINIMAX PARA TATETI")
    print("=" * 50)
    
    try:
        if verificar_implementacion():
            ejecutar_analisis()
            
            # Opcional: jugar contra la IA
            # jugar_vs_ia()
    except AssertionError as e:
        print(f"\n❌ Error en test: {e}")
        print("Revisá tu implementación y volvé a ejecutar.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Asegurate de implementar todas las funciones.")
