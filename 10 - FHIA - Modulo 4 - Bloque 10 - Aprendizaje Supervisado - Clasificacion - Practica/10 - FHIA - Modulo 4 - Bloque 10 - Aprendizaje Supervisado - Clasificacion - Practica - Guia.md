# 🧪 Guía de Trabajo — Clasificación y Métodos de Ensamble

## 🎯 Objetivo

Esta práctica no busca solo ejecutar código.

Busca que puedas:

- Entender cómo se comportan distintos modelos de clasificación
- Observar sus limitaciones
- Comparar resultados con métricas adecuadas
- Tomar decisiones según el problema

---

# 🟩 Parte A — Árboles de Decisión

## 🔍 Preguntas guía

- ¿Todos los modelos predicen lo mismo para el mismo caso?
- ¿La estructura del árbol cambia entre entrenamientos?
- ¿Qué provoca esos cambios?

## 🧠 Para pensar

- ¿El modelo parece estable o sensible a los datos?
- ¿Qué implica esto si lo llevamos a producción?

---

# 🟩 Parte B — Random Forest

## 🔍 Preguntas guía

- ¿Qué cambia respecto a la Parte A?
- ¿Las predicciones son más consistentes?
- ¿Qué efecto tiene combinar múltiples modelos?

## 🧠 Para pensar

- Si cada árbol individual es inestable…
- ¿por qué el conjunto es más estable?

---

# 🟩 Parte C — Evaluación de Modelos

## 🎯 Modelos a comparar

En esta parte vamos a comparar tres modelos de clasificación:

- Árbol de Decisión
- Random Forest
- XGBoost

El objetivo no es elegir automáticamente el que tenga el número más alto, sino justificar cuál conviene según el problema.

---

## 🔍 Preguntas iniciales

- ¿Qué modelo tiene mejor accuracy?
- ¿Eso es suficiente para decidir?

---

## ⚠️ Idea clave

> No todos los errores son iguales.

---

## 🚨 Escenario 1 — Prioridad: detectar todos los casos

- ¿Qué tipo de error es más grave?
- ¿Qué métrica es más relevante?
- ¿Qué modelo elegirías?

---

## 🚨 Escenario 2 — Prioridad: evitar falsas alarmas

- ¿Cambia el tipo de error importante?
- ¿Cambia la métrica?
- ¿Cambia el modelo elegido?

---

## 🔁 Sobre el threshold

- ¿Qué pasa al cambiar el umbral de decisión?
- ¿Cómo cambian precision y recall?

---

# 🧠 Reflexión Final

- ¿Elegiste el mismo modelo en todos los escenarios?
- ¿Existe un modelo “mejor” en general?
- ¿O depende del contexto?

---

# ⚠️ Errores comunes

- Elegir el modelo solo por accuracy
- No considerar el costo del error
- No analizar el efecto del threshold
- Asumir que un modelo siempre gana

---

# 🎯 Idea final

> Elegir un modelo no es aplicar una fórmula.  
> Es tomar una decisión en función del problema.