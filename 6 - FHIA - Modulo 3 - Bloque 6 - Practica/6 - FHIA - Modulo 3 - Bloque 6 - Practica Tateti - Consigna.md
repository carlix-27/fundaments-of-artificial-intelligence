# Práctica Bloque 5: Minimax para Tateti

**Curso:** Fundamentos e Historia de la Inteligencia Artificial  
**Universidad Austral**  
**Duración:** 40 minutos (en clase) + trabajo opcional en casa

---

## 🎯 Objetivos de Aprendizaje

Al completar esta práctica, podrás:

1. **Modelar un juego** como problema de búsqueda adversaria
2. **Implementar Minimax** desde cero
3. **Verificar optimalidad** comparando con resultados teóricos
4. **Medir eficiencia** contando nodos explorados
5. **[Extra]** Implementar poda Alfa-Beta y comparar rendimiento

---

## 📋 Contexto

El **Tateti** (Tic-Tac-Toe) es el juego perfecto para aprender búsqueda adversaria porque:

- Es **suficientemente simple** para analizar completamente (~5,500 estados únicos)
- Es **suficientemente complejo** para mostrar los conceptos (árbol de ~549,000 nodos sin optimizar)
- Tiene **solución conocida**: con juego perfecto, siempre termina en **empate**
- El **mejor primer movimiento** es el centro (posición 1,1)

---

## 🎮 Reglas del Tateti (recordatorio)

- Tablero 3×3, dos jugadores: **X** (MAX) y **O** (MIN)
- X siempre juega primero
- Gana quien logre 3 en línea (horizontal, vertical o diagonal)
- Si se llena el tablero sin ganador → **empate**

---

## 📐 Representación del Estado

Usaremos una matriz 3×3 donde:
- `' '` (espacio) = casilla vacía
- `'X'` = jugador MAX
- `'O'` = jugador MIN

```python
# Estado inicial
estado_inicial = [
    [' ', ' ', ' '],
    [' ', ' ', ' '],
    [' ', ' ', ' ']
]

# Ejemplo de estado intermedio (X jugó en centro, O en esquina)
estado_ejemplo = [
    ['O', ' ', ' '],
    [' ', 'X', ' '],
    [' ', ' ', ' ']
]
```

Las posiciones se indexan como `(fila, columna)`:
```
(0,0) | (0,1) | (0,2)
------+-------+------
(1,0) | (1,1) | (1,2)
------+-------+------
(2,0) | (2,1) | (2,2)
```

---

## 🔧 Funciones a Implementar

### 1. `acciones(estado)` → Lista de movimientos válidos

Retorna lista de tuplas `(fila, columna)` donde hay casillas vacías.

```python
def acciones(estado):
    """
    Retorna lista de acciones válidas (casillas vacías).
    
    Ejemplo:
        estado = [['X', ' ', ' '], [' ', 'O', ' '], [' ', ' ', ' ']]
        acciones(estado) → [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1), (2,2)]
    """
    # TODO: Implementar
    pass
```

### 2. `resultado(estado, accion, jugador)` → Nuevo estado

Retorna una **copia** del estado con la acción aplicada.

```python
def resultado(estado, accion, jugador):
    """
    Aplica la acción al estado y retorna el nuevo estado.
    ¡IMPORTANTE! No modificar el estado original, crear una copia.
    
    Ejemplo:
        estado = [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']]
        resultado(estado, (1, 1), 'X') → [[' ', ' ', ' '], [' ', 'X', ' '], [' ', ' ', ' ']]
    """
    # TODO: Implementar
    pass
```

### 3. `es_terminal(estado)` → bool

Retorna `True` si el juego terminó (alguien ganó o empate).

```python
def es_terminal(estado):
    """
    Verifica si el estado es terminal (victoria o empate).
    
    Ejemplo:
        estado = [['X', 'X', 'X'], ['O', 'O', ' '], [' ', ' ', ' ']]
        es_terminal(estado) → True (X ganó)
    """
    # TODO: Implementar
    pass
```

### 4. `utilidad(estado)` → int

Retorna el valor del estado terminal:
- `+1` si ganó X (MAX)
- `-1` si ganó O (MIN)
- `0` si empate

```python
def utilidad(estado):
    """
    Retorna la utilidad del estado terminal.
    Solo llamar si es_terminal(estado) == True
    
    Ejemplo:
        estado con X ganando → +1
        estado con O ganando → -1
        estado empate → 0
    """
    # TODO: Implementar
    pass
```

### 5. `minimax(estado, es_turno_max)` → int

El algoritmo principal. Retorna el valor minimax del estado.

```python
def minimax(estado, es_turno_max):
    """
    Algoritmo Minimax recursivo.
    
    Parámetros:
        estado: estado actual del juego
        es_turno_max: True si es turno de MAX (X), False si es MIN (O)
    
    Retorna:
        El valor minimax del estado (-1, 0, o +1)
    """
    # TODO: Implementar
    pass
```

### 6. `mejor_movimiento(estado)` → tupla

Encuentra el mejor movimiento para MAX desde el estado actual.

```python
def mejor_movimiento(estado):
    """
    Encuentra el mejor movimiento para X (MAX).
    
    Retorna:
        Tupla (fila, columna) del mejor movimiento
    """
    # TODO: Implementar
    pass
```

---

## 📊 Resultados Esperados

### Desde tablero vacío:

| Métrica | Valor Esperado |
|---------|----------------|
| Valor del juego | **0** (empate con juego perfecto) |
| Mejor movimiento | **Centro (1,1)** |
| Nodos explorados (Minimax) | **~549,946** |
| Tiempo aproximado | **2-5 segundos** |

### Con Alfa-Beta (extra):

| Métrica | Valor Esperado |
|---------|----------------|
| Valor del juego | **0** (idéntico) |
| Mejor movimiento | **Centro (1,1)** (idéntico) |
| Nodos explorados | **~5,000 - 50,000** |
| Tiempo aproximado | **0.05 - 0.5 segundos** |
| Mejora | **10-100× menos nodos** |

---

## ✅ Verificación

Usa esta función para verificar tu implementación:

```python
def verificar_implementacion():
    """Verifica que la implementación sea correcta."""
    
    # Test 1: Estado inicial
    estado = [[' ']*3 for _ in range(3)]
    valor = minimax(estado, True)
    mov = mejor_movimiento(estado)
    
    assert valor == 0, f"Error: valor debería ser 0, obtuve {valor}"
    assert mov == (1, 1), f"Error: mejor mov debería ser (1,1), obtuve {mov}"
    
    # Test 2: X puede ganar en un movimiento
    estado = [['X', 'X', ' '], ['O', 'O', ' '], [' ', ' ', ' ']]
    valor = minimax(estado, True)
    mov = mejor_movimiento(estado)
    
    assert valor == 1, f"Error: X puede ganar, valor debería ser 1"
    assert mov == (0, 2), f"Error: X debe jugar (0,2) para ganar"
    
    # Test 3: O ganó
    estado = [['X', 'X', 'O'], ['X', 'O', ' '], ['O', ' ', ' ']]
    assert es_terminal(estado) == True
    assert utilidad(estado) == -1
    
    print("✓ Todos los tests pasaron!")
```

---

## 📝 Entregables

### Obligatorio:
1. **Código funcional** con las 6 funciones implementadas
2. **Contador de nodos** explorados por Minimax
3. **Ejecución** mostrando:
   - Valor del juego desde estado inicial
   - Mejor primer movimiento
   - Cantidad de nodos explorados
   - Tiempo de ejecución

### Extra (bonus):
4. **Poda Alfa-Beta** implementada
5. **Comparación** de nodos explorados: Minimax vs Alfa-Beta
6. **Respuestas** a las preguntas de reflexión

---

## 🤔 Preguntas de Reflexión

Responde brevemente (2-3 oraciones cada una):

1. **¿Por qué Tateti siempre termina en empate con juego perfecto?**
   
2. **¿Cuántos estados únicos tiene Tateti?** (considerando simetrías: rotaciones y reflejos)

3. **Si quisieras escalar este código a Connect-4 (7×6, 4 en línea), ¿qué cambiarías?** ¿Funcionaría Minimax puro?

4. **[Si hiciste Alfa-Beta]** ¿Por qué el orden de los movimientos afecta la eficiencia de la poda?

---

## 💡 Tips de Implementación

### Tip 1: Copiar el estado correctamente
```python
# ❌ MAL - esto crea referencia, no copia
nuevo = estado

# ❌ MAL - copia superficial, las filas siguen siendo referencias
nuevo = estado.copy()

# ✅ BIEN - copia profunda
nuevo = [fila.copy() for fila in estado]
```

### Tip 2: Verificar ganador eficientemente
```python
# Líneas a verificar: 3 filas + 3 columnas + 2 diagonales = 8 líneas
lineas = [
    # Filas
    [(0,0), (0,1), (0,2)],
    [(1,0), (1,1), (1,2)],
    [(2,0), (2,1), (2,2)],
    # Columnas
    [(0,0), (1,0), (2,0)],
    [(0,1), (1,1), (2,1)],
    [(0,2), (1,2), (2,2)],
    # Diagonales
    [(0,0), (1,1), (2,2)],
    [(0,2), (1,1), (2,0)]
]
```

### Tip 3: Contar nodos
```python
# Usar variable global o pasar como parámetro
nodos_explorados = 0

def minimax(estado, es_turno_max):
    global nodos_explorados
    nodos_explorados += 1
    # ... resto del algoritmo
```

### Tip 4: Estructura de Alfa-Beta
```python
def alfabeta(estado, es_turno_max, alfa, beta):
    """
    alfa: mejor valor que MAX puede garantizar (piso)
    beta: mejor valor que MIN puede garantizar (techo)
    
    Si beta <= alfa → PODAR (no explorar más hijos)
    """
    # TODO: Implementar
    pass

# Llamada inicial:
valor = alfabeta(estado_inicial, True, float('-inf'), float('inf'))
```

---

## 🎯 Criterios de Éxito

| Criterio | Puntos |
|----------|--------|
| `acciones()` correcto | 10 |
| `resultado()` correcto (con copia) | 10 |
| `es_terminal()` correcto | 15 |
| `utilidad()` correcto | 10 |
| `minimax()` correcto | 25 |
| `mejor_movimiento()` correcto | 10 |
| Contador de nodos | 5 |
| Código limpio y documentado | 5 |
| **Subtotal** | **90** |
| **[Extra]** Alfa-Beta | +15 |
| **[Extra]** Comparación con métricas | +5 |
| **[Extra]** Preguntas de reflexión | +10 |
| **Total máximo** | **120** |

---

## 📚 Referencias

- Russell & Norvig, *Artificial Intelligence: A Modern Approach*, Capítulo 5
- [Wikipedia: Tic-tac-toe](https://en.wikipedia.org/wiki/Tic-tac-toe) - Análisis de complejidad
- [Minimax Algorithm (GeeksforGeeks)](https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-1-introduction/)

---

**¡Buena suerte! 🎮**
