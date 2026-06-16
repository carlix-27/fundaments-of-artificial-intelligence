# Bloque 8: Tarea
# Construcción de un Clasificador de Spam Naive Bayes desde Cero

## Objetivo

En esta tarea vas a construir manualmente un modelo simple de clasificación de texto.

El objetivo es que comprendas:

- cómo se estiman probabilidades a partir de datos
- cómo se combinan para tomar decisiones
- qué significa "aprender" en Machine Learning

**No se permite el uso de librerías ni modelos preentrenados. Todo debe resolverse a partir de los datos provistos.**

---

# Dataset

Se dispone del siguiente conjunto de emails etiquetados:

| Email | Clase | Texto |
|---------|---------|---------|
| 1 | spam | "oferta gratis urgente" |
| 2 | spam | "gana dinero rapido" |
| 3 | spam | "oferta exclusiva limitada" |
| 4 | spam | "urgente gana dinero ahora" |
| 5 | spam | "promocion especial gratis" |
| 6 | spam | "dinero facil rapido" |
| 7 | spam | "oferta limitada ahora" |
| 8 | spam | "gana dinero facil" |
| 9 | legit | "reunion mañana proyecto" |
| 10 | legit | "agenda reunión equipo" |
| 11 | legit | "avance del proyecto final" |
| 12 | legit | "reunión mañana oficina" |
| 13 | legit | "entrega final proyecto" |
| 14 | legit | "planificación equipo proyecto" |
| 15 | legit | "reunión de seguimiento" |
| 16 | legit | "agenda semanal equipo" |

---

# Palabras a considerar

Para simplificar el problema, se trabajará únicamente con el siguiente conjunto de palabras:

```text
{gratis, oferta, urgente, dinero, gana, proyecto, reunión}