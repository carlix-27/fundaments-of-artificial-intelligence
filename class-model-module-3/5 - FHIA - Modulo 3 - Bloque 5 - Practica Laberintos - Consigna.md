# Práctica Bloque 5: Pathfinding en Laberintos

**Curso:** Fundamentos e Historia de la Inteligencia Artificial  
**Universidad Austral**

---

## 🎯 Objetivos

- Implementar BFS, DFS y A* a partir del pseudocódigo y el starter provisto
- Comparar su eficiencia (nodos expandidos)
- Entender el impacto de las heurísticas

---

## 🧠 Idea clave (leer antes de empezar)

Los tres algoritmos que vas a implementar comparten la misma estructura.

Todos:
- toman un nodo de una frontera
- verifican si es el objetivo
- expanden vecinos

👉 **Lo único que cambia es cómo eligen el próximo nodo a explorar.**

| Algoritmo | Estrategia |
|----------|-----------|
| BFS | FIFO (cola) |
| DFS | LIFO (pila) |
| A* | Menor f(n) = g(n) + h(n) |

---

## ✍️ Antes de programar (obligatorio)

Responder brevemente:

1. ¿Qué partes del algoritmo son idénticas en BFS y DFS?
2. ¿Qué única línea cambia?
3. ¿Qué información adicional usa A* que los otros no usan?

👉 Pista: mirá el `while`.

---

## 🧩 Problema

Se te da un laberinto representado como una matriz:

- `0` = celda libre
- `1` = pared

### Definiciones

- Estado: `(fila, columna)`
- Start: `(0, 0)`
- Goal: `(n-1, n-1)`
- Acciones: Norte, Sur, Este, Oeste
- Costo: 1 por movimiento

---

## 🧪 Tarea

Implementar:

- BFS
- DFS
- A*

A partir del pseudocódigo visto en clase.

El starter te da:
- estructura base
- funciones auxiliares
- laberintos

👉 Tu tarea es completar la lógica.

---

## 📊 Qué debes medir

Para cada algoritmo:

- Longitud del camino encontrado
- Cantidad de nodos expandidos

---

## 📈 Resultados esperados

### Laberinto 15×15

| Algoritmo | Camino | Nodos expandidos |
|----------|--------|------------------|
| BFS | 29 | ~150–180 |
| DFS | Variable | Variable |
| A* | 29 | ~60–80 |

---

### Laberinto 25×25

| Algoritmo | Camino | Nodos expandidos |
|----------|--------|------------------|
| BFS | 49 | ~400–450 |
| DFS | Variable | Variable |
| A* | 49 | ~120–160 |

---

## 📝 Informe (½–1 página)

Responder:

1. ¿Por qué A* expande menos nodos que BFS?
2. ¿Qué observaste con DFS?
3. ¿Qué pasaría si la heurística no fuera admisible?
4. ¿En qué casos preferirías DFS sobre BFS?

---

## 🏁 Mensaje final

👉 No estás implementando tres algoritmos distintos.  
👉 Estás implementando **la misma idea con distintas estrategias de exploración**.