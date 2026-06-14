# Bloque 13 — Fundamentos de Redes Neuronales
### Material de práctica

---

## 1. Guía de práctica para estudiantes

# Práctica Bloque 13 — Redes Neuronales en Acción

## Introducción

En esta práctica vamos a experimentar con redes neuronales simples para desarrollar intuición sobre:

- capas ocultas
- no-linealidad
- entrenamiento
- overfitting
- generalización

La práctica **NO busca construir sistemas industriales complejos.**

El objetivo es entender: **cómo aprende una red neuronal.**

---

## Parte A — TensorFlow Playground

**Acceso:** https://playground.tensorflow.org

---

### Experimento 1 — XOR

**Objetivo:** Comprender por qué una sola neurona no puede resolver XOR.

**Pasos:**

1. Seleccionar dataset **XOR**.
2. Configurar: **0 hidden layers**.
3. Entrenar.
4. Observar la frontera de decisión.
5. Agregar: **1 hidden layer, 2 neuronas**.
6. Reentrenar.

**Preguntas:**

1. ¿Por qué falla la red sin capa oculta?
2. ¿Qué cambia cuando agregamos neuronas ocultas?
3. ¿La solución es lineal o no lineal?

**Respuestas:**

1. Falla porque XOR no es separable linealmente. Sin capas ocultas, la red solo puede aprender una frontera lineal y no puede separar correctamente los cuatro casos del problema.
2. Al agregar neuronas ocultas, la red puede combinar transformaciones intermedias y construir una frontera no lineal que sí resuelve XOR.
3. La solución es no lineal, porque necesita una transformación del espacio de entrada antes de poder separar las clases.

---

### Experimento 2 — Activaciones

**Objetivo:** Comparar distintas funciones de activación.

**Pasos:**

1. Dataset **Circle**.
2. 1 hidden layer con 4 neuronas.
3. Comparar: **Linear, Sigmoid, Tanh, ReLU**.

**Preguntas:**

1. ¿Qué ocurre con activación lineal?
2. ¿Cuál converge más rápido?
3. ¿Por qué ReLU es tan usada?

**Respuestas:**

1. Con activación lineal, toda la red se comporta como un modelo lineal, aunque tenga varias capas. Entonces pierde la capacidad de modelar fronteras complejas.
2. En general ReLU suele converger más rápido porque es simple, no satura tan fácil y mantiene gradientes útiles en más rango de valores que sigmoid.
3. ReLU se usa mucho porque es eficiente, ayuda a entrenar redes profundas y suele funcionar bien en la práctica para aprender patrones no lineales.

---

### Experimento 3 — Learning Rate

**Objetivo:** Observar cómo afecta el learning rate al entrenamiento.

**Pasos:**

1. Dataset **Gaussian**.
2. 2 hidden layers.
3. Probar learning rate: **muy pequeño, muy grande, razonable**.

**Preguntas:**

1. ¿Qué ocurre con learning rate pequeño?
2. ¿Qué ocurre con learning rate grande?
3. ¿Qué valor parece más estable?

**Respuestas:**

1. Con learning rate pequeño, el entrenamiento avanza muy lento y puede quedarse corto dentro del número de épocas disponible.
2. Con learning rate grande, los pasos son demasiado bruscos: la pérdida puede oscilar, divergir o no estabilizarse.
3. El valor más estable suele ser uno intermedio, suficientemente grande para avanzar pero no tan grande como para romper la optimización.

---

### Experimento 4 — Overfitting

**Objetivo:** Observar cómo una red puede memorizar el training set.

**Pasos:**

1. Dataset **Spiral**.
2. Reducir training ratio.
3. Crear una red muy grande.
4. Entrenar.
5. Observar: **training loss** y **test loss**.
6. Activar regularización **L2**.

**Preguntas:**

1. ¿Cómo identificas overfitting?
2. ¿Qué cambia con regularización?
3. ¿Por qué minimizar training loss no alcanza?

**Respuestas:**

1. Se identifica cuando el modelo rinde muy bien en training pero peor en test, y además la frontera de decisión se vuelve demasiado irregular o ajustada al ruido.
2. Con regularización, la red tiende a aprender pesos más pequeños y una frontera más suave, lo que reduce la tendencia a memorizar.
3. Porque un training loss bajo no garantiza generalización. El objetivo real es rendir bien en datos nuevos, no solo en los vistos durante el ajuste.

## Respuestas al notebook de la Parte 13

### 4. Modelo lineal: una frontera no alcanza

1. ¿La frontera lineal captura bien la estructura del dataset?

No. Captura solo una parte del patrón y deja errores porque `moons` tiene una estructura no lineal.

2. ¿Qué tipo de errores comete?

Comete errores de separación geométrica: mezcla regiones que pertenecen a distintas clases y no sigue la forma curva del dataset.

3. ¿Por qué este ejemplo se parece conceptualmente al problema XOR?

Porque en ambos casos las clases no se pueden separar con una sola recta. Hace falta una transformación no lineal para resolverlos.

### 5. Red neuronal simple

1. ¿Qué cambió respecto del modelo lineal?

La red puede construir fronteras curvas y adaptarse a estructuras no lineales, por lo que mejora claramente la separación.

2. ¿La frontera parece más flexible?

Sí. La frontera deja de ser rígida y puede seguir mejor la geometría del problema.

3. ¿Dirías que la red está memorizando o aprendiendo un patrón general?

Si train y test quedan cercanos, parece más un aprendizaje general que una simple memorización.

4. ¿Dónde ves la idea de “representación aprendida”?

En que las capas ocultas transforman las coordenadas de entrada en una representación más útil para separar clases.

### 6. Actividad 1 — Cambiar arquitectura

1. ¿Qué arquitectura produjo la frontera más simple?

La arquitectura más pequeña, por ejemplo `(4,)`, suele producir la frontera más simple.

2. ¿Qué arquitectura produjo la frontera más compleja?

Las redes más grandes, como `(64, 64, 64, 64)`, suelen generar fronteras más complejas.

3. ¿Más neuronas siempre mejoran la generalización?

No. Más capacidad puede ayudar al ajuste, pero también aumentar el riesgo de overfitting.

4. ¿Qué señales mirarías para sospechar overfitting?

Diferencia grande entre train y test, frontera demasiado enredada y sensibilidad excesiva al ruido.

### 7. Actividad 2 — Cambiar función de activación

1. ¿Qué activación produjo mejor resultado?

Normalmente `relu` o `tanh` dan mejores resultados que `logistic` en este tipo de problema, aunque depende de la corrida y los hiperparámetros.

2. ¿Cuál pareció entrenar más lento?

`logistic` suele entrenar más lento por saturación de gradientes; `tanh` también puede ser más lenta que `relu`.

3. ¿Por qué una activación no lineal es necesaria?

Porque sin no linealidad las capas se colapsan en una sola transformación lineal y la red pierde expresividad.

4. ¿Qué pasaría si todas las capas fueran lineales?

La red completa sería equivalente a un modelo lineal simple, sin importar cuántas capas tuviera.

### 8. Actividad 3 — Learning rate

1. ¿Qué ocurre con learning rate muy pequeño?

El entrenamiento progresa muy lentamente y puede quedarse lejos de una buena solución.

2. ¿Qué ocurre con learning rate muy grande?

La optimización se vuelve inestable: la loss puede subir y bajar sin converger bien.

3. ¿El learning rate cambia la capacidad de representación o la optimización?

La optimización. La capacidad de representación depende de la arquitectura y la activación; el learning rate afecta cómo se llega a una solución.

4. ¿Cómo se conecta esto con la metáfora de “bajar la montaña”?

Un paso chico avanza lento; un paso grande puede pasarse del valle. El learning rate controla el tamaño de cada movimiento en esa bajada.

### 9. Provocar overfitting

1. ¿La frontera parece más irregular?

Sí. Una red grande con pocos datos suele dibujar una frontera muy ajustada a las muestras disponibles.

2. ¿Hay diferencia entre accuracy de train y test?

Sí, normalmente el train sube mucho más que el test.

3. ¿Qué evidencia sugiere overfitting?

La brecha entre train y test, más una frontera demasiado específica para el ruido del conjunto pequeño.

4. ¿Qué relación hay entre capacidad del modelo y riesgo de memorización?

Cuanta más capacidad tiene el modelo, más fácil le resulta memorizar detalles del entrenamiento en lugar de aprender patrones generales.

### 10. Regularización L2

1. ¿La frontera se volvió más suave?

Sí, en general L2 suaviza la frontera porque penaliza pesos grandes.

2. ¿Cambió la diferencia entre train y test?

Suele reducirse, aunque a veces el train baja un poco porque el modelo deja de sobreajustar.

3. ¿La regularización mejoró la generalización?

Normalmente sí, porque obliga al modelo a buscar soluciones menos extremas.

4. ¿Por qué penalizar pesos grandes puede ayudar?

Porque pesos pequeños suelen producir funciones más estables y menos sensibles al ruido.

### 11. Comparación final

1. ¿Por qué un modelo lineal no alcanza para `moons`?

Porque la estructura del problema es curvada y no puede separarse con una sola recta.

2. ¿Qué aportan las capas ocultas?

Permiten construir transformaciones intermedias que vuelven separable un problema no lineal.

3. ¿Qué rol cumple la función de activación?

Introduce no linealidad. Sin activación no hay composición útil de capas.

4. ¿Qué diferencia hay entre capacidad de representación y optimización?

La capacidad de representación es qué tan complejo puede ser el modelo; la optimización es qué tan bien logramos entrenarlo.

5. ¿Cómo identificaste overfitting?

Por la brecha entre train y test y por la frontera demasiado compleja frente a pocos datos.

6. ¿Qué efecto tuvo la regularización?

Disminuyó el sobreajuste y suavizó la solución.

7. ¿Qué significa que una red “aprenda una representación”?

Que convierte los datos originales en otra forma interna más útil para resolver la tarea.

8. ¿Qué conexión ves entre esta práctica y TensorFlow Playground?

La misma intuición: explorar cómo cambian frontera, capacidad, activaciones, learning rate y regularización al modificar la red.