# Bloque 11 — Práctica
## Descubrimiento de regiones latentes en el Diagrama HR con Gaia

**Materia:** Fundamentos e Historia de la Inteligencia Artificial  
**Tema:** Aprendizaje no supervisado  
**Dataset:** `gaia_stars_sample_14k.csv`  
**Modalidad:** el código ya está provisto. La tarea consiste en **ejecutar, observar, interpretar y criticar** el resultado.

---

## 1. Contexto

En esta práctica trabajamos con una muestra balanceada de **14.000 estrellas reales de Gaia DR3**, distribuidas en siete clases espectrales:

- O
- B
- A
- F
- G
- K
- M

La columna `spectral_class` existe en el dataset, pero **no se usa para entrenar el algoritmo**. Se revela recién al final para comparar la estructura encontrada por K-Means con las clases espectrales conocidas.

La pregunta central no es:

> ¿Podemos clasificar estrellas con accuracy supervisada?

La pregunta correcta es:

> ¿Puede un algoritmo no supervisado descubrir regiones interpretables del Diagrama HR usando color y magnitud absoluta?

---

## 2. Variables centrales

El ejercicio se apoya en el plano clásico del Diagrama HR:

| Variable | Significado |
|---|---|
| `bp_rp` | Color fotométrico: `BPmag - RPmag` |
| `GMAG` | Magnitud absoluta en banda G |

En el Diagrama HR, el eje horizontal representa color/temperatura aproximada y el eje vertical representa brillo intrínseco. Por convención astronómica, en los gráficos se invierte el eje de magnitud: las estrellas más luminosas aparecen arriba.

También se incluyen variables auxiliares para interpretación posterior:

- `Teff`: temperatura efectiva
- `Rad`: radio estimado
- `Lum-Flame`: luminosidad estimada
- `Plx`: paralaje
- `Dist`: distancia estimada

Estas variables ayudan a interpretar los clusters, pero el núcleo del clustering se basa en `bp_rp` y `GMAG`.

---

## 3. Qué hace el pipeline

El script `Bloque_11_Practica_Gaia_HR_KMeans.py` realiza los siguientes pasos:

1. Carga el dataset Gaia.
2. Construye o verifica la variable `bp_rp = BPmag - RPmag`.
3. Usa solo `bp_rp` y `GMAG` como variables principales del plano HR.
4. Estandariza las variables con `StandardScaler`.
5. Ejecuta K-Means para `K = 2 ... 10`.
6. Calcula inercia, coeficiente de silueta y pureza posterior.
7. Genera gráficos del codo, silueta y pureza posterior.
8. Grafica el Diagrama HR coloreado por clusters.
9. Grafica el Diagrama HR coloreado por clase espectral real.
10. Compara clusters contra clases espectrales mediante tablas `cluster × spectral_class`.
11. Genera perfiles físicos promedio por cluster.
12. Aplica PCA sobre el espacio HR como visualización complementaria.

---

## 4. Decisiones didácticas importantes

### 4.1. K-Means no ve la clase espectral

Durante el clustering, K-Means no sabe qué estrellas son O, B, A, F, G, K o M. Solo ve puntos en el plano:

```text
bp_rp, GMAG
```

La clase espectral se usa solo al final, como comparación posterior.

### 4.2. K sugerido por el algoritmo vs K del dominio

El script calcula un `K` sugerido por máxima silueta. Ese K puede no ser 7.

También se genera una comparación usando:

```text
K = 7
```

porque existen siete clases espectrales en el dataset.

Esto permite discutir una idea central del aprendizaje no supervisado:

> El número de clusters geométricamente conveniente no tiene por qué coincidir con el número de categorías humanas o científicas.

### 4.3. Pureza posterior no es accuracy

El script calcula una pureza posterior contra `spectral_class`. Esa pureza **no es accuracy supervisada**. K-Means nunca optimizó esa etiqueta.

La pureza sirve solamente para preguntar:

> ¿Hasta qué punto la partición geométrica encontrada por K-Means se parece a las clases espectrales reales?

---

## 5. Entregables

El informe del estudiante debe incluir:

### 5.1. Selección de K

Responder:

1. ¿Qué K sugiere la silueta?
2. ¿Dónde aparece el codo en la inercia?
3. ¿Tiene sentido comparar con K=7? ¿Por qué?

Incluir el gráfico:

```text
01_k_selection_elbow_silhouette_purity.png
```

---

### 5.2. Diagrama HR por clusters

Analizar los gráficos:

```text
02_hr_clusters_k_silhouette.png
03_hr_clusters_k7.png
```

Responder:

1. ¿Los clusters aparecen como islas separadas?
2. ¿O parecen cortes sobre regiones continuas del Diagrama HR?
3. ¿Qué zonas del HR ocupa cada cluster?

---

### 5.3. Comparación contra clase espectral real

Analizar:

```text
04_hr_spectral_class_real.png
cluster_vs_spectral_class_counts_*.csv
cluster_vs_spectral_class_normalized_*.csv
```

Responder:

1. ¿K-Means recupera claramente OBAFGKM?
2. ¿Qué clases se mezclan más?
3. ¿Qué clases parecen más separables?
4. ¿Qué significa que un cluster contenga varias clases espectrales?

---

### 5.4. Interpretación física de clusters

Usar:

```text
05_cluster_profiles_k7.png
cluster_profiles_k7.csv
```

Responder:

1. ¿Qué clusters tienen mayor temperatura promedio?
2. ¿Qué clusters tienen mayor luminosidad promedio?
3. ¿Qué clusters corresponden a estrellas más rojas o más azules?
4. ¿Los clusters son interpretables físicamente?

---

### 5.5. Reflexión final

Responder en 8–12 líneas:

> ¿K-Means descubrió clases espectrales reales o impuso una partición geométrica sobre el Diagrama HR? Justificá la respuesta.

La respuesta esperada no debe ser ingenuamente positiva ni negativa. La idea clave es:

> El aprendizaje no supervisado puede revelar estructura latente, pero el significado de esa estructura no viene dado automáticamente por el algoritmo.

---

## 6. Archivos generados por el script

El script crea una carpeta:

```text
bloque11_gaia_hr_outputs
```

con gráficos, tablas y el dataset enriquecido con clusters.

---

## 7. Idea conceptual central

Esta práctica muestra una tensión fundamental:

> K-Means agrupa por geometría en el espacio de features. Las clases espectrales son categorías físicas/humanas. Ambas estructuras pueden estar relacionadas, pero no tienen por qué coincidir perfectamente.

Esa diferencia es una de las ideas más importantes del aprendizaje no supervisado.
