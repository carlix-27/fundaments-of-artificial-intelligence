# Respuestas conceptuales · Práctica Bloque 14

Este archivo resume el análisis pedido en [preguntas.md](preguntas.md) y lo fundamenta con una ejecución real del script [Bloque 14 - Practica Hogarena.py](Bloque%2014%20-%20Practica%20Hogarena.py).

## 0. Ejecución y evidencia

Se ejecutó el script en modo rápido con salida en [bloque14_outputs_quick](bloque14_outputs_quick).

Archivos generados:

- [A_fashion_mnist_samples.png](bloque14_outputs_quick/A_fashion_mnist_samples.png)
- [A_mlp_vs_cnn_accuracy.png](bloque14_outputs_quick/A_mlp_vs_cnn_accuracy.png)
- [A_mlp_vs_cnn_loss.png](bloque14_outputs_quick/A_mlp_vs_cnn_loss.png)
- [B_time_series_sample.png](bloque14_outputs_quick/B_time_series_sample.png)
- [B_predictions_seq20.png](bloque14_outputs_quick/B_predictions_seq20.png)
- [B_training_seq20.png](bloque14_outputs_quick/B_training_seq20.png)
- [B_predictions_seq50.png](bloque14_outputs_quick/B_predictions_seq50.png)
- [B_training_seq50.png](bloque14_outputs_quick/B_training_seq50.png)
- [C_attention_matrix_toy.png](bloque14_outputs_quick/C_attention_matrix_toy.png)
- [resultados_bloque14.json](bloque14_outputs_quick/resultados_bloque14.json)

## 1. Resultados empíricos

### Parte A: MLP vs CNN sobre Fashion-MNIST

| Modelo | Parámetros | Test accuracy | Test loss | Tiempo |
| --- | ---: | ---: | ---: | ---: |
| MLP | 109,386 | 0.8420 | 0.4548 | 3.3 s |
| CNN | 220,234 | 0.8633 | 0.3841 | 21.3 s |

Interpretación: la CNN obtuvo mejor accuracy y menor loss que el MLP. El costo fue mayor tiempo de entrenamiento, algo esperable porque el modelo convolucional trabaja con una estructura espacial más rica y más operaciones por batch.

### Parte B: RNN vs LSTM para series temporales

| Secuencia | Modelo | Parámetros | MSE | MAE |
| --- | --- | ---: | ---: | ---: |
| 20 | RNN | 1,633 | 0.0233 | 0.1223 |
| 20 | LSTM | 4,897 | 0.0298 | 0.1409 |
| 50 | RNN | 1,633 | 0.0203 | 0.1158 |
| 50 | LSTM | 4,897 | 0.0272 | 0.1335 |

Interpretación: en este dataset sintético corto, la RNN simple resultó mejor que la LSTM. Eso no contradice la teoría: muestra que la ventaja de LSTM aparece sobre todo cuando el contexto largo es realmente relevante o cuando el problema hace sufrir más al gradiente. En esta práctica rápida, el patrón de la serie es suficientemente suave para que la RNN compita bien.

### Parte C: self-attention toy

La matriz de atención resultante fue de tamaño $5 \times 5$ y cada fila sumó 1.

Rasgos observables:

- El primer token concentró gran parte de su atención en `perro`.
- El último token (`perro`) se atendió casi por completo a sí mismo, con peso cercano a 1.
- La atención no es uniforme: hay selectividad contextual, no simple promedio.

## 2. Parte A — CNN vs MLP

### Respuestas

1. La CNN aprovecha la estructura espacial local de la imagen: vecindad de pixeles, bordes, texturas y patrones jerárquicos. El MLP la aplana y pierde esa noción explícita de cercanía.

2. El *weight sharing* reutiliza el mismo filtro en múltiples posiciones. Eso reduce el número de patrones distintos que el modelo debe aprender por ubicación y agrega un sesgo inductivo de invariancia traslacional.

3. Cuando se mezclan los píxeles de forma consistente en train y test, se destruye la estructura espacial local. La CNN pierde gran parte de su ventaja. Eso es importante porque muestra que la superioridad convolucional depende de que la arquitectura esté alineada con el dato.

4. La CNN no es más inteligente; está mejor alineada con datos tipo imagen por su sesgo inductivo: localidad, compartición de pesos y composición jerárquica.

### Sustento con el run

La CNN superó al MLP en accuracy de test ($0.8633$ vs $0.8420$) y obtuvo menor loss ($0.3841$ vs $0.4548$). La diferencia no se explica solo por capacidad, porque el MLP también tenía más de $10^5$ parámetros; la clave está en que la CNN explota la geometría de la imagen.

## 3. Parte B — RNN vs LSTM

### Respuestas

1. En teoría, la ventaja de LSTM aumenta al crecer `seq_length` porque preserva mejor la memoria de largo plazo y mitiga *vanishing gradient*. En esta corrida rápida, la RNN simple quedó por encima en ambas longitudes, lo que sugiere que el problema no exigía dependencias tan largas como para justificar el costo extra de LSTM.

2. *Forget gate* decide qué conservar o borrar; *input gate* decide qué información nueva entra; *output gate* controla qué parte de la memoria se expone como salida.

3. Una RNN simple puede ser suficiente cuando las dependencias temporales son cortas, el patrón es suave y se busca menor costo computacional.

4. Las LSTM siguen siendo secuenciales y por eso motivan a los Transformers: mejor paralelismo y conexión más directa entre posiciones lejanas.

### Sustento con el run

Con `seq_length=20`, la RNN obtuvo MSE $0.0233$ y MAE $0.1223$, mientras que la LSTM quedó en MSE $0.0298$ y MAE $0.1409$. Con `seq_length=50`, la RNN también fue mejor: MSE $0.0203$ frente a $0.0272$. La lectura correcta no es “RNN siempre gana”, sino “la arquitectura más compleja no siempre aporta si el patrón temporal no lo necesita”.

## 4. Parte C — Self-attention / QKV

### Respuestas

1. Q y K producen los pesos de atención porque la similitud entre lo que un token busca y lo que otro ofrece determina relevancia.

2. Lo que se combina al final son los Values porque ahí está el contenido que se quiere agregar al contexto.

3. Una fila de la matriz $A$ representa la distribución de atención de un token query sobre todos los tokens key.

4. Hace falta *positional encoding* porque la self-attention pura no tiene orden intrínseco.

5. El Transformer evita el cuello de botella secuencial de RNN/LSTM y mejora el paralelismo.

### Sustento con el run

El experimento toy mostró una matriz $5 \times 5$ donde cada fila suma 1, como exige la normalización softmax. Además, la atención fue claramente selectiva: por ejemplo, la primera fila asignó aproximadamente $0.84$ a `perro`, y la última fila casi toda la masa a sí misma. Eso ilustra que la atención no promedia indiscriminadamente; pondera contexto de forma aprendida.

## 5. Parte D — IA híbrida y RAG

### Respuestas

1. Un LLM puro puede fallar porque alucina, no garantiza vigencia normativa y puede confundir reglamentos o fechas.

2. Un sistema RAG aporta grounding: recupera documentos reales antes de generar y mejora trazabilidad.

3. Componentes mínimos:

- Modelo: LLM generativo.
- Base documental: reglamentos, planes y calendario versionado.
- Recuperación: índice léxico y/o semántico.
- Ranking: reranker.
- Generación: prompt con contexto recuperado.
- Herramientas: consultas estructuradas.
- Memoria: contexto conversacional de corto plazo.
- Control: validaciones, políticas y fallback.

4. Persisten riesgos como recuperación incompleta, documentos desactualizados, interpretación errónea y sobreconfianza.

5. El sistema debería mostrar fuentes, secciones, versión, fragmentos utilizados y, si corresponde, advertencias de ambigüedad.

## 6. Parte E — Discriminativo vs generativo

### Respuestas

1. MLP y CNN sobre Fashion-MNIST son discriminativos porque aprenden $p(y|x)$: predicen etiqueta a partir de entrada.

2. En RAG, el rol generativo lo cumple el LLM que redacta la respuesta final.

3. La diferencia depende principalmente del objetivo de entrenamiento, no solo de la arquitectura.

4. BERT y GPT muestran que una misma familia arquitectural puede servir para distintos objetivos: BERT suele usarse de forma discriminativa o representacional, GPT de forma generativa autoregresiva.

## 7. Reflexión final

Para imagen elegiría CNN cuando el dato preserve estructura espacial local, porque su sesgo inductivo coincide con el problema. Para series temporales con dependencias cortas usaría RNN simple por costo; para dependencias largas preferiría LSTM o un Transformer temporal si la escala lo justifica. Para texto largo, un Transformer con atención es la opción natural por paralelismo y manejo de contexto amplio. Para sistemas con conocimiento externo, usaría una arquitectura híbrida tipo RAG: recuperación + reranking + generación con fuentes. En todos los casos, la elección no depende solo de potencia bruta, sino de alineación entre arquitectura y estructura del dato. Ese ajuste entre sesgo inductivo y tarea suele explicar mejor el rendimiento y la robustez que simplemente aumentar parámetros.

## 8. Cierre

La conclusión que deja la práctica es la que pide [preguntas.md](preguntas.md): no existe una arquitectura universalmente superior. La pregunta correcta es qué inductive bias coincide mejor con el problema. CNN para imagen, RNN/LSTM para temporalidad, self-attention para dependencias globales, y RAG para sistemas que necesitan conocimiento externo verificable.