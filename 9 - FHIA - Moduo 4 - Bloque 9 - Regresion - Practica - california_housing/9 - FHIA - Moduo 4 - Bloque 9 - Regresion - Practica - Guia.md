# Bloque 9 — Guía del Mini Lab
## Aprendizaje Supervisado: Regresión

### Mini Lab: Regresión Lineal como Diagnóstico de Modelos

---

## 1. Objetivo del lab

En esta práctica no solo vas a implementar regresión lineal. Vas a usarla para entender cómo piensa un ingeniero de Machine Learning.

El objetivo no es obtener un número, sino responder:

> ¿El modelo aprendió algo útil?  
> ¿Generaliza o memoriza?  
> ¿Qué decisión tomaría a partir de las métricas?

---

## 2. Archivos necesarios

Para realizar la práctica necesitás tener en la misma carpeta:

1. `bloque9_regresion_starter.py`
2. `california_housing.csv`

El archivo `california_housing.csv` ya está provisto.  
No hace falta descargar datos desde internet.

La solución docente está en:

- `bloque9_regresion_solucion.py`

---

## 3. Requisitos de entorno

Necesitás Python 3.10+ y las siguientes librerías:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Si usás Anaconda, también podés ejecutar la práctica desde un entorno donde esas librerías ya estén instaladas.

Importante: ejecutá el script desde una carpeta donde tengas permisos de escritura y lectura.  
No lo ejecutes directamente desde `C:\`.

---

## 4. Dataset

Trabajaremos con el dataset **California Housing**:

- 20.640 observaciones
- 8 variables de entrada
- variable objetivo: precio mediano de vivienda (`MedHouseVal`)

El dataset se carga localmente desde:

```python
df = pd.read_csv("california_housing.csv")
```

---

## 5. Qué vamos a implementar

Implementaremos dos formas de entrenar regresión lineal:

1. **Ecuaciones normales**  
   Solución analítica directa.

2. **Gradient Descent**  
   Solución iterativa basada en optimización.

Ambos métodos buscan lo mismo:

> encontrar los parámetros `w` que minimizan el MSE.

---

## 6. Estructura conceptual

La práctica sigue el mismo framework visto en clase:

1. **Datos + Modelo**
2. **Función de pérdida**
3. **Optimización**
4. **Evaluación**
5. **Diagnóstico**

La parte importante es el paso 5.

No alcanza con entrenar. Hay que interpretar.

---

## 7. Parte 1 — Baseline

### Qué hacemos

Antes de entrenar un modelo, construimos un baseline:

> predecir siempre el promedio del training set.

### Por qué importa

El baseline responde:

> ¿Qué tan bien puedo hacerlo sin Machine Learning?

Si nuestro modelo no mejora al baseline, el modelo no sirve.

### Pregunta de decisión

**¿El modelo de regresión lineal mejora claramente al baseline?**

---

## 8. Parte 2 — Ecuaciones normales

### Qué hacemos

Implementamos la solución analítica:

```text
w* = (X^T X)^(-1) X^T y
```

En código usaremos una forma numéricamente más estable:

```python
w = np.linalg.pinv(X) @ y
```

### Qué significa

Encontramos directamente los parámetros óptimos para regresión lineal.

### Pregunta de decisión

**¿Tiene sentido usar una solución analítica en datasets grandes?**

Pista: pensá en memoria, costo computacional y escalabilidad.

---

## 9. Parte 3 — Gradient Descent

### Qué hacemos

Implementamos un algoritmo iterativo:

1. Predecir con los pesos actuales
2. Calcular error
3. Calcular gradiente
4. Actualizar pesos
5. Repetir

### Qué significa

Gradient Descent no es el objetivo del aprendizaje.  
Es un mecanismo para minimizar la pérdida.

El objetivo sigue siendo:

> generalizar bien a datos nuevos.

### Pregunta de decisión

**¿Qué pasa si la tasa de aprendizaje `alpha` es demasiado grande o demasiado chica?**

---

## 10. Parte 4 — Train vs Test

### Qué hacemos

Calculamos MSE en:

- Training set
- Test set

### Qué significa

Comparar ambos errores permite diagnosticar generalización.

### Reglas de interpretación

| Situación | Diagnóstico |
|---|---|
| Train bajo, test mucho más alto | Posible overfitting |
| Train alto y test alto | Posible underfitting |
| Train y test bajos y cercanos | Buen ajuste relativo |

### Pregunta de decisión

**¿Tu modelo generaliza o solo ajusta bien el training set?**

---

## 11. Parte 5 — Comparación de métodos

### Qué hacemos

Comparamos:

- baseline
- ecuaciones normales
- gradient descent

según:

- MSE train
- MSE test
- tiempo de entrenamiento
- estabilidad
- escalabilidad

### Pregunta de decisión

**¿Cuál método elegirías en este caso? ¿Y cuál elegirías si tuvieras millones de datos?**

No alcanza con decir “el que da menor MSE”.

---

## 12. Parte 6 — Experimentos con alpha

### Qué hacemos

Probamos distintos valores de `alpha`:

- 0.001
- 0.01
- 0.1

### Qué buscamos observar

- Convergencia lenta
- Convergencia estable
- Oscilación o divergencia

### Pregunta de decisión

**¿Cómo sabés si el entrenamiento está funcionando?**

Pista: mirá la curva de MSE.

---

## 13. Entregables obligatorios

Tu entrega debe incluir:

1. **Código completo ejecutado** (notebook o `.py`)
2. **Outputs relevantes**, incluyendo:
   - MSE de train y test para cada método
   - tabla comparativa
   - gráfico de convergencia de Gradient Descent
3. **Respuestas a las preguntas de análisis** de la sección siguiente

---

## 14. Análisis y preguntas de decisión

Responde de forma clara y justificada. No alcanza con copiar resultados.

### 1. Diagnóstico del modelo

Reporta:

- MSE Train (Ecuaciones Normales)
- MSE Test (Ecuaciones Normales)

Luego responde:

- ¿El modelo está en **underfitting, overfitting o buen ajuste**?
- Justifica tu respuesta usando la diferencia entre train y test.

No respondas con definiciones. Usá tus resultados.

---

### 2. Comparación de métodos

Compara:

- Ecuaciones normales
- Gradient Descent

Responde:

- ¿Llegan al mismo resultado? ¿Por qué?
- ¿Cuál fue más rápido en este caso?
- ¿Cuál elegirías si el dataset tuviera millones de datos?

Explica en términos de **optimización y escalabilidad**.

---

### 3. Efecto de la tasa de aprendizaje (`alpha`)

Describe qué ocurrió al usar:

- `alpha = 0.001`
- `alpha = 0.01`
- `alpha = 0.1`

Responde:

- ¿Cuál converge más lento?
- ¿Cuál es inestable o diverge?
- ¿Cómo detectarías que Gradient Descent está funcionando correctamente?

Usá el gráfico de MSE para justificar.

---

### 4. Interpretación del aprendizaje

Responde:

- ¿Qué significa realmente “aprender” en este modelo?
- ¿Qué rol cumple el MSE en ese proceso?
- ¿Qué está optimizando el algoritmo?

---

### 5. Pregunta conceptual anti-IA

Supón que obtenés:

- MSE Train = 0.20
- MSE Test = 0.65

Responde:

- ¿Qué está pasando?
- ¿Qué acciones concretas tomarías para mejorar el modelo?

Justifica usando conceptos vistos en clase: generalización, complejidad, regularización, datos, etc.

---

## 15. Condición de entrega

- No se aceptan respuestas sin justificación.
- No se aceptan respuestas puramente teóricas sin referencia a los resultados obtenidos.
- Se evaluará la coherencia entre código, resultados y análisis.

---

## 16. Criterios de evaluación

| Criterio | Peso |
|---|---:|
| Código correcto y ejecutable | 30% |
| Resultados coherentes | 20% |
| Diagnóstico del modelo e interpretación | 30% |
| Claridad y calidad de las explicaciones | 20% |

---

## 17. Nivel de completitud

- **Entrega completa:** incluye Gradient Descent funcionando.
- **Entrega parcial aceptable:** incluye hasta Ecuaciones Normales + análisis correcto.

Se prioriza **comprensión sobre implementación completa**.

---

## 18. Idea central de la tarea

> Machine Learning no es solo entrenar modelos.  
> Es interpretar resultados y tomar decisiones basadas en datos.

Si tu código corre pero no sabés explicar qué pasó, la tarea no está completa.

---

## 19. Cierre

Después de esta práctica deberías poder responder:

- ¿Cuándo un modelo está funcionando bien?
- ¿Cómo detectar overfitting?
- ¿Por qué necesitamos métodos como Gradient Descent?
- ¿Qué significa realmente “aprender de los datos”?
