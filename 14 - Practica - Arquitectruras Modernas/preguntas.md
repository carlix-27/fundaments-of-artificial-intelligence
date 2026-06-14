# Práctica Hogareña · Bloque 14

## Arquitecturas modernas, atención e IA híbrida

> **Idea central:** la práctica no busca demostrar que una arquitectura sea universalmente superior. Busca justificar por qué una arquitectura funciona mejor cuando su sesgo inductivo coincide con la estructura del dato.

---

# 1. Objetivos de aprendizaje

* Comparar empíricamente MLP y CNN sobre imágenes, interpretando localidad espacial, convolución, pooling y *weight sharing*.
* Comparar SimpleRNN y LSTM sobre series temporales, observando memoria selectiva, longitud de secuencia y *vanishing gradient*.
* Explicar *self-attention* mediante Q, K, V, *scores*, *softmax*, matriz de atención y salida contextual.
* Distinguir objetivos discriminativos y generativos, y separar arquitectura de objetivo de entrenamiento.
* Analizar por qué un sistema moderno puede combinar modelo, recuperación externa, herramientas, memoria y control.
* Justificar la elección de arquitectura según estructura del dato, restricciones del problema y riesgos del sistema.

---

# 2. Modalidad y entregables

| Elemento                | Qué entregar                                                                                                                           |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Código ejecutado        | Script `.py` o notebook con outputs visibles. Puede usarse modo *quick* si la computadora es lenta.                                    |
| Gráficos                | MLP vs CNN, RNN vs LSTM, matriz de atención *toy* y experimento de píxeles mezclados.                                                  |
| Respuestas conceptuales | Máximo 2 páginas. Responder con precisión y conexión con la clase, no con definiciones largas.                                         |
| Reflexión final         | Un párrafo de 5 a 8 oraciones: qué arquitectura elegirías para imagen, serie temporal, texto largo y sistema con conocimiento externo. |

---

# 3. Preparación técnica y ejecución

**Requisitos mínimos:** Python 3.9+, TensorFlow 2.x, NumPy, Matplotlib y scikit-learn.

En caso de problemas locales, ejecutar en Google Colab.

## Instalación sugerida

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Modo recomendado para entregar si no tenés GPU

```bash
python "Bloque 14 - Practica Hogarena.py" --quick
```

## Modo completo (más lento)

```bash
python "Bloque 14 - Practica Hogarena.py" --full
```

## Elegir carpeta de salida

```bash
python "Bloque 14 - Practica Hogarena.py" --quick --out mis_resultados
```

---

# 4. Archivos esperados de salida

| Archivo                            | Uso en el informe                                                |
| ---------------------------------- | ---------------------------------------------------------------- |
| `A_fashion_mnist_samples.png`      | Muestra del dataset de imágenes.                                 |
| `A_mlp_vs_cnn_accuracy.png`        | Comparación del entrenamiento MLP vs CNN.                        |
| `A_mlp_vs_cnn_loss.png`            | Comparación del entrenamiento MLP vs CNN.                        |
| `B_time_series_sample.png`         | Serie temporal sintética usada para predicción.                  |
| `B_predictions_seq20.png`          | Comparación visual RNN vs LSTM.                                  |
| `B_predictions_seq50.png`          | Comparación visual RNN vs LSTM.                                  |
| `B_training_seq20.png`             | Curvas de entrenamiento para secuencias cortas y medias.         |
| `B_training_seq50.png`             | Curvas de entrenamiento para secuencias cortas y medias.         |
| `C_attention_matrix_toy.png`       | Matriz de atención toy: filas como queries y columnas como keys. |
| `resultados_bloque14.json`         | Métricas y resultados numéricos guardados por el script.         |
| `plantilla_respuestas_bloque14.md` | Plantilla editable para ordenar las respuestas conceptuales.     |

---

# 5. Parte A — CNN vs MLP para imágenes

**Dataset:** Fashion-MNIST.

Comparar un MLP *fully-connected* contra una CNN simple. Ambos modelos reciben la misma información visual, pero la CNN codifica estructura espacial mediante convoluciones, pooling y reutilización de filtros.

| Modelo                    | Estructura                                               | Qué mirar                                                       |
| ------------------------- | -------------------------------------------------------- | --------------------------------------------------------------- |
| MLP                       | Flatten → Dense → Dense → Softmax                        | Trata la imagen como vector; pierde vecindad espacial.          |
| CNN                       | Conv2D → Pool → Conv2D → Pool → Dense → Softmax          | Explota localidad espacial y patrones repetidos.                |
| CNN con píxeles mezclados | Misma CNN sobre imágenes con posición espacial destruida | Sirve para probar cuándo el sesgo convolucional deja de ayudar. |

## Preguntas de análisis

1. ¿Qué estructura de la imagen aprovecha la CNN que el MLP ignora?
2. ¿Por qué *weight sharing* reduce grados de libertad aunque el número total de parámetros dependa de la implementación?
3. ¿Qué ocurre cuando se mezclan aleatoriamente los píxeles de forma consistente en train y test? ¿Por qué ese resultado es importante?
4. ¿La CNN es "más inteligente" que el MLP, o está mejor alineada con el dato? Justificá.

---

# 6. Parte B — RNN vs LSTM para series temporales

**Dataset sintético:** combinación de ondas sinusoidales con ruido.

Comparar SimpleRNN y LSTM usando distintas longitudes de secuencia.

| Experimento                | Configuración mínima | Qué observar                                                                             |
| -------------------------- | -------------------- | ---------------------------------------------------------------------------------------- |
| Secuencia corta            | `seq_length = 20`    | La RNN simple puede competir si el contexto relevante es corto.                          |
| Secuencia media            | `seq_length = 50`    | La LSTM suele estabilizar mejor el entrenamiento y reducir error.                        |
| Secuencia larga (opcional) | `seq_length = 100`   | La ventaja de LSTM debería hacerse más visible, aunque depende de datos y entrenamiento. |

## Preguntas de análisis

1. ¿La ventaja de LSTM aumenta al aumentar `seq_length`? ¿Qué relación tiene esto con *vanishing gradient*?
2. ¿Qué función conceptual cumplen *forget gate*, *input gate* y *output gate*?
3. ¿En qué casos una RNN simple podría ser suficiente?
4. ¿Qué problema estructural de las LSTM motiva luego a los Transformers?

---

# 7. Parte C — Atención escalada y Q/K/V

No se pide implementar un Transformer completo. Se pide explicar el mecanismo central de *self-attention* a partir de sus fórmulas.

[
Q = XW_Q
]

[
K = XW_K
]

[
V = XW_V
]

[
S = \frac{QK^T}{\sqrt{d_k}}
]

[
A = softmax(S)
]

[
Z = AV
]

[
z_j = \sum_i \alpha_{j,i}V_i
]

## Preguntas de análisis

1. ¿Por qué Q y K producen los pesos de atención?
2. ¿Por qué lo que se combina finalmente son los Values?
3. ¿Qué representa una fila de la matriz A?
4. ¿Por qué hace falta *positional encoding* si usamos *self-attention*?
5. ¿Qué cuello de botella de RNN/LSTM evita el Transformer?

---

# 8. Parte D — IA híbrida y RAG

**Caso:** un chatbot universitario debe responder preguntas sobre reglamentos internos actualizados, correlatividades, programas de materias y fechas administrativas.

## Preguntas de análisis

1. ¿Por qué un LLM puro puede fallar en este caso?
2. ¿Qué aportaría un sistema RAG?
3. Identificá los componentes mínimos del sistema:

   * Modelo
   * Base documental
   * Recuperación
   * Ranking
   * Generación
   * Herramientas
   * Memoria
   * Control
4. ¿Qué riesgos persisten aunque el sistema use RAG?
5. ¿Qué evidencia o trazabilidad debería mostrar el sistema para ser confiable?

---

# 9. Parte E breve — Discriminativo vs generativo

Esta parte cubre explícitamente la distinción entre arquitectura y objetivo de entrenamiento.

## Preguntas

1. ¿Por qué MLP y CNN sobre Fashion-MNIST son modelos discriminativos?
2. ¿Qué parte de un sistema RAG cumple el rol generativo?
3. ¿La diferencia entre discriminativo y generativo depende de la arquitectura o del objetivo de entrenamiento?
4. Usá BERT/GPT o CNN/Diffusion como ejemplo de que una familia arquitectural puede servir para distintos objetivos.

---

# 10. Estructura recomendada del informe

1. Resultados Parte A: MLP vs CNN.
2. Resultados Parte B: RNN vs LSTM.
3. Explicación Parte C: Q/K/V y matriz de atención.
4. Análisis Parte D: RAG e IA híbrida.
5. Parte E: discriminativo vs generativo.
6. Reflexión final integradora.

---

# 11. Rúbrica de evaluación

| Criterio                         | Peso | Descripción                                                                                          |
| -------------------------------- | ---- | ---------------------------------------------------------------------------------------------------- |
| Código ejecutado y resultados    | 25%  | El script corre, genera gráficos y reporta métricas. No se exige coincidencia numérica exacta.       |
| Interpretación arquitectura-dato | 30%  | Relaciona resultados con sesgo inductivo, estructura espacial, temporal y acceso global al contexto. |
| Atención / QKV                   | 20%  | Explica correctamente Q, K, V, scores, softmax, matriz A y salida contextual.                        |
| IA híbrida / RAG                 | 15%  | Analiza límites del DL puro y justifica recuperación, herramientas, memoria y control.               |
| Discriminativo vs generativo     | 10%  | Distingue objetivo de entrenamiento de familia arquitectural.                                        |

---

# 12. Criterio de aprobación conceptual

Una respuesta correcta no se limita a reportar "CNN ganó" o "LSTM tuvo menor MSE".

Debe explicar por qué la arquitectura elegida se alinea o no con la estructura del dato.

**El foco de la práctica es la interpretación.**
