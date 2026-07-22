# Bloque 15 — Auditoría y Despliegue Responsable de Sistemas de IA

> Módulo 6 — IA Responsable: impacto, riesgo y gobernanza
> Cierre transversal del curso "Fundamentos e Historia de la IA"

---

## 1. Idea central del bloque

**Tesis:** un modelo no es responsable solo porque predice bien. Predecir bien es *necesario, pero no suficiente*. La responsabilidad no es una capa moral que se agrega después: es una propiedad del sistema completo en su contexto de despliegue.

El bloque marca un pivot conceptual:

| Hasta acá (preguntas técnicas) | Desde acá (preguntas sistémicas) |
|---|---|
| ¿Alcanza la métrica objetivo? | ¿A quiénes afecta y cómo? |
| ¿Generaliza a datos nuevos? | ¿Es auditable? |
| ¿Converge el entrenamiento? | ¿Quién responde por sus errores? |
| ¿Es eficiente en cómputo? | ¿Debe existir bajo esta forma? |

El sistema técnico no cambia. Cambia la pregunta que le hacemos.

Tres preguntas que estructuran todo el bloque:
1. ¿Es justo en su efecto sobre distintos grupos? → **Fairness**
2. ¿Es explicable a quien afecta? → **Explicabilidad**
3. ¿Debería desplegarse bajo estas condiciones? → **Riesgo y gobernanza**

---

## 2. Fairness y sesgo

### 2.1 Por qué una métrica global no alcanza
Una accuracy global del 92% puede esconder un desempeño de 78% en un subgrupo (con FNR de 0.35 vs 0.04 en el grupo mayoritario). **La métrica global no es falsa, es incompleta.** Toda auditoría seria debe desagregar por subgrupo (accuracy, precision, recall, FPR, FNR).

### 2.2 El sesgo atraviesa todo el pipeline, no solo el dataset
1. **Datos**: sesgos históricos, sub-representación, medición defectuosa.
2. **Diseño y modelo**: variables proxy, feature engineering, función objetivo/regularización.
3. **Despliegue**: feedback loops (el sistema modifica el mundo que después usa para reentrenarse — la fuente más peligrosa), distribution shift, contexto de uso imprevisto.

### 2.3 Casos reales
- **COMPAS (2016, justicia penal EE.UU.)**: reincidencia predicha con FPR 2x superior en un grupo. Mostró que con tasas base distintas, *calibración* y *paridad de errores* pueden entrar en conflicto matemático irreducible.
- **Amazon Hiring (2018)**: modelo entrenado sobre CVs históricos penalizó marcadores asociados a género femenino. Se discontinuó.
- **Gender Shades (2018, Joy Buolamwini/MIT)**: clasificadores comerciales de género en imágenes: 99% de accuracy en hombres de piel clara vs 65% en mujeres de piel oscura.

### 2.4 Tres criterios de fairness (mutuamente incompatibles en general)

| Criterio | Definición | Trade-off |
|---|---|---|
| **Paridad demográfica** | P(Ŷ=1\|A=a) ≈ P(Ŷ=1\|A=b) | Ignora la distribución real del fenómeno |
| **Paridad de errores (equalized odds)** | FPᵃ≈FPᵇ y FNᵃ≈FNᵇ | Requiere acceso a la etiqueta verdadera por grupo |
| **Calibración por grupo** | P(Y=1\|score=s,A=g)≈s ∀g | Puede coexistir con disparidad en tasas/errores |

**No existe una única métrica de equidad universal.** Con tasas base distintas entre grupos, calibración y paridad de errores son matemáticamente incompatibles salvo casos especiales (resultado teórico clave, similar al de Kleinberg/Chouldechova).

### 2.5 Caso conductor: scoring crediticio (ejemplo cuantitativo de auditoría)
- **Regla del 80%** (umbral EEOC 1978): Tasa B / Tasa A < 0.80 → disparidad. Ejemplo: 0.65 → FALLA.
- **Gap absoluto**: diferencia > 10pp en tasa de aprobación. Ejemplo: 23pp → FALLA.
- **ΔFPR / ΔFNR**: disparidad > 0.05 entre grupos. Ejemplo: ΔFNR=0.18 → FALLA.

### 2.6 Variables proxy: la trampa silenciosa
Excluir la variable sensible (género, etnia, nivel socioeconómico) **no excluye el sesgo** si quedan features correlacionadas: código postal → nivel socioeconómico; historial laboral → género (lagunas por maternidad); tipo de empleo → nivel socioeconómico; ingresos → género/etnia. El modelo discrimina sin "saberlo".

**Implicancia operativa:**
1. Auditar correlaciones entre features incluidas y atributos sensibles excluidos.
2. Medir disparidad de salidas por grupo aunque el grupo no sea input directo.
3. Considerar *fairness through awareness* en lugar de *fairness through blindness*.

---

## 3. Interpretabilidad y explicabilidad

### 3.1 El problema de la caja negra
A mayor capacidad/complejidad del modelo, mayor dificultad de explicación directa:
- Regresión logística (~12 coef.): inspeccionable a ojo, capacidad moderada.
- Random Forest (~100 árboles): importancia agregada sí, razón individual no.
- Transformer/LLM (10⁹–10¹² parámetros): caja negra estructural, ni sus creadores trazan la razón de una salida puntual.

### 3.2 Interpretabilidad ≠ Explicabilidad post-hoc

| | Interpretabilidad | Explicabilidad post-hoc |
|---|---|---|
| Qué es | Propiedad del modelo (transparente por diseño) | Herramienta externa aplicada sobre un modelo opaco |
| Ejemplos | Regresión lineal/logística, árboles chicos, reglas IF-THEN, GAMs | LIME, SHAP, feature importance, contrafactuales |
| Ventaja | Auditable sin herramientas externas | Aplicable a modelos arbitrariamente complejos |
| Límite | Capacidad predictiva limitada | Es una aproximación, no la razón real |

### 3.3 LIME (Local Interpretable Model-agnostic Explanations)
Aproxima *localmente* un modelo opaco con un modelo simple:
1. Tomar una predicción individual (instancia x).
2. Perturbar la instancia (generar variantes sintéticas alrededor de x).
3. Observar al modelo opaco (pedirle predicción sobre cada variante).
4. Ajustar un modelo simple local (regresión lineal ponderada por cercanía a x).

Responde: *"para este caso, ¿qué variables empujaron la predicción?"* — no describe el comportamiento global.

### 3.4 SHAP (SHapley Additive exPlanations)
Distribuye la predicción entre features a partir de un valor base (predicción promedio): ŷ(x) = φ₀ + Σφᵢ(x). Basado en valores de Shapley (teoría de juegos cooperativos).
- Uso local: qué pesó positivo/negativo en un caso.
- Uso global: promediar |φᵢ| sobre el dataset → importancia global.
- **Límite**: contribución predictiva ≠ causalidad. SHAP describe el modelo, no el mundo.

### 3.5 Explicar no es justificar
- Explicar ≠ correcto (descriptiva, no validante)
- Explicar ≠ justo (un proxy explicado sigue siendo un proxy)
- Explicar ≠ aceptable (transparencia no resuelve si debe usarse en ese contexto)
- Explicar ≠ legitimizar (una buena explicación puede volver defendible lo que no debería desplegarse)

### 3.6 Límites de XAI post-hoc
No prueban causalidad · son aproximaciones · pueden ser inestables (cambia la explicación según muestra/vecindad/método) · dan falsa sensación de control. **Complementan auditoría, testing y supervisión humana — no los reemplazan.**

---

## 4. Seguridad y robustez (eje ortogonal a fairness/XAI)

Un sistema justo y explicable puede seguir siendo inseguro:

1. **Ataques adversariales**: perturbaciones imperceptibles que fuerzan errores de clasificación.
2. **Data poisoning**: inyección de datos maliciosos durante entrenamiento.
3. **Membership inference**: inferir si un dato formó parte del training set → fuga de información sensible.
4. **Prompt injection**: manipulación de modelos generativos para alterar instrucciones o filtrar contenido.
5. **Distribution shift**: degradación silenciosa cuando producción se aleja de entrenamiento.

La auditoría cubre **todo el ciclo de vida**, no solo la evaluación inicial.

---

## 5. Mitigación y trade-offs

### 5.1 Cuándo intervenir

| Etapa | Técnicas |
|---|---|
| **Pre-procesamiento** (datos) | Recolección dirigida, reweighting, resampling, revisión de proxies, balanceo de subgrupos |
| **En-procesamiento** (entrenamiento) | Restricciones de fairness en la función objetivo, penalización de disparidad, adversarial debiasing, regularización orientada a equidad |
| **Post-procesamiento** (decisión) | Ajuste de umbrales, calibración posterior, opción de rechazo, revisión humana, monitoreo de disparidad en producción |

**No existe mitigación gratuita**: cada intervención tiene costo en otra dimensión.

### 5.2 Trade-offs estructurales
- Fairness ↔ performance (redistribuye errores, no solo los reduce)
- Paridad ↔ frontera de decisión (mueve umbral/FP/FN)
- Interpretabilidad ↔ capacidad (transparencia by design recorta arquitecturas)
- Mitigación ↔ complejidad operativa (pipelines de auditoría = costo de ingeniería)

---

## 6. Gobernanza basada en riesgo

### 6.1 Por qué la técnica sola no alcanza
1. La métrica no fija el umbral aceptable — cuánta disparidad es tolerable es una decisión **institucional**, no técnica.
2. Los incentivos privados pueden divergir del bien público (quien desarrolla absorbe beneficios, externaliza costos).
3. A escala, los errores son sistémicos: un modelo desplegado replica la misma lógica sobre miles/millones de decisiones.

> La técnica detecta y mide. La gobernanza decide qué hacer con lo medido.

### 6.2 Principio organizador
A mayor daño potencial, mayor nivel de control exigido — no según complejidad técnica, sino según **contexto de uso**.

### 6.3 EU AI Act (marco por niveles de riesgo)

| Nivel | Régimen | Ejemplos |
|---|---|---|
| Inaceptable | Prohibido | Social scoring estatal, manipulación subliminal, categorización biométrica sensible, policía predictiva por perfilado |
| Alto riesgo | Evaluación de conformidad | Salud, justicia, empleo/RRHH, educación, crédito y servicios esenciales |
| Riesgo limitado | Transparencia obligatoria | Chatbots (informar que es IA), deepfakes etiquetados, generación de contenido |
| Riesgo mínimo | Sin obligaciones específicas | Filtros de spam, IA en videojuegos, recomendación de productos |

### 6.4 Requisitos de ingeniería para sistemas de alto riesgo
Gestión de riesgos continua · gobernanza de datos (calidad/trazabilidad) · documentación técnica (model cards) · logging automático · transparencia al usuario · supervisión humana real (comprender, cuestionar, revertir) · robustez y ciberseguridad · accountability (responsabilidades exigibles). **No es compliance, es arquitectura.**

### 6.5 NIST AI RMF (marco voluntario, no regulación)
Cuatro funciones cíclicas:
- **Govern**: cultura, roles, responsabilidades (atraviesa a las otras tres).
- **Map**: contexto de uso, stakeholders, impactos posibles.
- **Measure**: métricas de performance, fairness, robustez, seguridad.
- **Manage**: priorizar, mitigar, monitorear en producción de forma continua.

> EU AI Act dice **qué** exige la ley; NIST AI RMF propone **cómo** organizarlo como proceso de ingeniería.

### 6.6 GDPR y decisiones automatizadas
No regula la IA como tal, sino el tratamiento de datos personales, perfilado y decisiones automatizadas.
- **No** existe un "derecho absoluto y genérico a explicación" (formulación imprecisa habitual).
- **Sí** exige: base legal, transparencia sobre la lógica, salvaguardas ante decisiones con efectos significativos (Art. 22).
- Implicancias de ingeniería: intervención humana significativa (no formal), mecanismos de impugnación, trazabilidad, explicación accionable.

### 6.7 Tres preguntas independientes
1. **Técnica**: ¿funciona? (accuracy, robustez, cobertura)
2. **Legal**: ¿está permitido? (GDPR, EU AI Act, regulación sectorial)
3. **Moral**: ¿debería implementarse? (dignidad, proporcionalidad, efectos institucionales)

Cumplir la ley no garantiza actuar éticamente. Funcionar técnicamente no garantiza ser aceptable. **Las tres deben cumplirse, son independientes entre sí.**

### 6.8 Decisión profesional final: tres salidas posibles
- **✓ Desplegar**: auditoría aprobada, supervisión definida, documentación completa, monitoreo activo.
- **≈ Condicionar**: alcance reducido, supervisión reforzada, revisión periódica, plan de salida.
- **✕ Detener**: rediseño, recolección de datos, cambio de arquitectura, reformular el problema.

---

## 7. Guía de implementación práctica

Esta sección traduce la teoría en pasos concretos para armar un pipeline de auditoría real (útil tanto para el TP integrador como para cualquier proyecto profesional).

### 7.1 Pipeline de auditoría técnica (Parte A — scoring / clasificación)

```
1. EDA por subgrupo
   - Distribución de la variable objetivo por grupo protegido
   - Tasas base por grupo (esto determina qué tensiones de fairness vas a tener)

2. Métricas de performance desagregadas
   - Accuracy, precision, recall, F1 por grupo (no solo global)
   - Matriz de confusión por grupo
   - Librerías: sklearn.metrics + groupby, o fairlearn.metrics

3. Métricas de fairness
   - Demographic parity ratio (regla del 80%) → fairlearn.metrics.demographic_parity_ratio
   - Equalized odds: ΔFPR y ΔFNR → fairlearn.metrics.equalized_odds_difference
   - Definir umbrales de aceptación ANTES de correr el análisis (evita motivated reasoning)

4. Detección de proxies
   - Correlación (point-biserial / Cramér's V) entre features incluidas y atributo sensible excluido
   - Si hay correlación alta → documentar como proxy y decidir tratamiento

5. Explicabilidad
   - SHAP global (summary plot) → qué pesa en general
   - SHAP/LIME local sobre casos frontera y sobre casos con error de tipo distinto por grupo
   - Cruzar: ¿los proxies detectados en el paso 4 aparecen con peso alto en SHAP?

6. Mitigación (elegir 1, discutir otra)
   - Pre: reweighting con fairlearn.preprocessing o resampling manual
   - En: ExponentiatedGradient de fairlearn (restricción de fairness en el training)
   - Post: ThresholdOptimizer de fairlearn (ajuste de umbral por grupo)
   - Reportar el trade-off: cuánto bajó la disparidad vs cuánto bajó accuracy/AUC global

7. Informe final
   - Tabla comparativa antes/después de mitigación
   - Recomendación cuantificada (¿pasa los 3 criterios: regla 80%, gap absoluto, ΔFPR/ΔFNR?)
```

**Stack sugerido:** Python, `scikit-learn`, `fairlearn`, `shap`, `lime`, `pandas`, `matplotlib/seaborn` para las tablas comparativas por grupo.

### 7.2 Pipeline de auditoría de riesgo y gobernanza (Parte B — triaje / alto riesgo)

No es un pipeline de código sino un **framework de decisión documentado**. Estructura sugerida del informe:

```
1. Clasificación de riesgo (EU AI Act)
   - ¿En qué categoría cae el sistema? (justificar con el dominio: salud = alto riesgo)
   - Consecuencia: qué obligaciones aplican (evaluación de conformidad, documentación, etc.)

2. Mapeo de impacto (NIST AI RMF - Map)
   - Stakeholders: pacientes, médicos, hospital, aseguradoras
   - Magnitud del daño en el peor caso (falso negativo en triaje = muerte evitable)
   - Asimetría del daño: ¿un tipo de error es mucho peor que el otro? (esto define
     dónde poner el umbral, más allá de la métrica óptima en ROC)

3. Supervisión humana requerida (Art. 22 GDPR + EU AI Act alto riesgo)
   - Definir el nivel: ¿human-in-the-loop (aprueba cada decisión),
     human-on-the-loop (supervisa y puede intervenir), o human-in-command (override total)?
   - Para triaje hospitalario: casi siempre human-in-the-loop obligatorio,
     el modelo prioriza, el médico decide

4. Explicabilidad y documentación requerida
   - Nivel de explicación que necesita el médico (no el data scientist):
     accionable, en el momento de la decisión, no un dashboard de SHAP values
   - Trazabilidad: logging de cada decisión, versión de modelo, inputs usados

5. Encuadre regulatorio + decisión final
   - Checklist de los 8 requisitos de ingeniería para alto riesgo (sección 6.4)
   - Aplicar las 3 preguntas independientes (técnica / legal / moral)
   - Recomendación final: Desplegar / Condicionar / Detener, con justificación explícita
     de qué condición dispara cada opción
```

### 7.3 Checklist reutilizable para cualquier proyecto real

Antes de deployar cualquier sistema de ML/IA con impacto sobre personas:

- [ ] ¿Medí performance desagregada por subgrupo, no solo global?
- [ ] ¿Identifiqué qué criterio de fairness prioriza el dominio (paridad demográfica vs paridad de errores vs calibración) y por qué?
- [ ] ¿Busqué variables proxy correlacionadas con atributos sensibles excluidos?
- [ ] ¿Apliqué XAI (SHAP/LIME) sobre casos de error, no solo sobre casos exitosos?
- [ ] ¿Distingo explícitamente "explicado" de "justo" y de "aceptable"?
- [ ] ¿Evalué robustez ante distribution shift y ataques adversariales, más allá del set de test?
- [ ] ¿Clasifiqué el sistema según nivel de riesgo (EU AI Act) o el marco regulatorio que aplique?
- [ ] ¿Definí el nivel de supervisión humana real (no formal)?
- [ ] ¿Hay logging/trazabilidad suficiente para auditar una decisión puntual después del hecho?
- [ ] ¿La decisión final (desplegar/condicionar/detener) está justificada en los tres ejes —técnico, legal, moral— y no solo en el técnico?

---

## 8. Cierre conceptual

> La inteligencia artificial no elimina la responsabilidad humana. La desplaza, la amplifica y la vuelve más urgente.

El recorrido del bloque cierra el círculo del curso: de "¿puede una máquina pensar?" (Turing, 1950 — pregunta filosófica) a "¿qué hacemos cuando una máquina decide?" (2026 — pregunta de ingeniería y de sociedad, sobre qué condiciones de despliegue, supervisión y responsabilidad aceptamos).