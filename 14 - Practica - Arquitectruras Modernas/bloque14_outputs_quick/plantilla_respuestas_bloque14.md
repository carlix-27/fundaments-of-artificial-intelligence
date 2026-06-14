# Respuestas conceptuales · Práctica Bloque 14

## Parte A — CNN vs MLP

1. ¿Qué estructura de la imagen aprovecha la CNN que el MLP ignora?

2. ¿Por qué weight sharing reduce grados de libertad aunque el número total de parámetros dependa de la implementación?

3. ¿Qué ocurre cuando se mezclan aleatoriamente los píxeles de forma consistente en train y test? ¿Por qué ese resultado es importante?

4. ¿La CNN es “más inteligente” que el MLP, o está mejor alineada con el dato?

## Parte B — RNN vs LSTM

1. ¿La ventaja de LSTM aumenta al aumentar seq_length? ¿Qué relación tiene esto con vanishing gradient?

2. ¿Qué función conceptual cumplen forget gate, input gate y output gate?

3. ¿En qué casos una RNN simple podría ser suficiente?

4. ¿Qué problema estructural de las LSTM motiva luego a los Transformers?

## Parte C — Self-attention / QKV

1. ¿Por qué Q y K producen los pesos de atención?

2. ¿Por qué lo que se combina finalmente son los Values?

3. ¿Qué representa una fila de la matriz A?

4. ¿Por qué hace falta positional encoding si usamos self-attention?

5. ¿Qué cuello de botella de RNN/LSTM evita el Transformer?

## Parte D — IA híbrida y RAG

Caso: chatbot universitario sobre reglamentos, correlatividades, programas y fechas administrativas.

1. ¿Por qué un LLM puro puede fallar en este caso?

2. ¿Qué aportaría un sistema RAG?

3. Identificá los componentes mínimos: modelo, base documental, recuperación, ranking, generación, herramientas, memoria y control.

4. ¿Qué riesgos persisten aunque el sistema use RAG?

5. ¿Qué evidencia o trazabilidad debería mostrar el sistema para ser confiable?

## Parte E — Discriminativo vs generativo

1. ¿Por qué MLP y CNN sobre Fashion-MNIST son modelos discriminativos?

2. ¿Qué parte de un sistema RAG cumple el rol generativo?

3. ¿La diferencia entre discriminativo y generativo depende de la arquitectura o del objetivo de entrenamiento?

4. Usá BERT/GPT o CNN/Diffusion como ejemplo de que una familia arquitectural puede servir para distintos objetivos.

## Reflexión final

En 5 a 8 oraciones: ¿qué arquitectura elegirías para imagen, serie temporal, texto largo y sistema con conocimiento externo? Justificá con la idea de sesgo inductivo y arquitectura de sistema.
