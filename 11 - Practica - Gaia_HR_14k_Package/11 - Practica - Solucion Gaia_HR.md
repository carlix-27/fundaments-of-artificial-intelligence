# Bloque 11 — Solución y Análisis
## Descubrimiento de regiones latentes en el Diagrama HR con Gaia

### 1. Selección de K

El criterio de silueta sugiere **K = 5**, porque alcanza el valor máximo del barrido K=2..10. En la curva de inercia también se ve una caída fuerte hasta K≈5 y luego una mejora más lenta, así que el codo aparece aproximadamente en esa zona.

Comparar con **K = 7** sí tiene sentido, porque el dataset está balanceado por siete clases espectrales OBAFGKM. Esa comparación no busca optimizar K-Means, sino contrastar la partición geométrica con una estructura física conocida.

### 2. Qué muestran los clusters en el HR

Los clusters no aparecen como islas completamente separadas. Más bien cortan regiones continuas del Diagrama HR.

Con **K = 7**, la separación visual es más interpretable que con K = 5, aunque sigue habiendo mezcla en zonas de transición. El HR recupera bien la estructura general: secuencia principal, estrellas azules y luminosas, y ramas rojas/frías.

### 3. Comparación contra la clase espectral real

La tabla de conteos para **K = 7** muestra que varios clusters sí se alinean bastante con clases espectrales dominantes, pero no de forma perfecta:

- **Cluster 2** está dominado por **M**.
- **Cluster 3** está dominado por **M** y también tiene bastante **K**.
- **Cluster 5** concentra sobre todo **A** y **B**.
- **Cluster 0** agrupa sobre todo **F** y **G**.
- **Cluster 1** mezcla bastante **B** con **O**.

La tabla normalizada confirma que hay mezcla importante entre clases vecinas, especialmente en las fronteras físicas del HR. Eso es esperable: K-Means busca proximidad geométrica, no etiquetas astronómicas.

La **pureza posterior** para **K = 7** es aproximadamente **0.50**, mientras que para **K = 5** es menor. Eso significa que K = 7 se parece más a la partición espectral, aunque no sea el mejor K por silueta.

### 4. Interpretación física de los clusters

Los perfiles físicos de **K = 7** muestran un patrón coherente:

- Los clusters con **menor bp_rp** son más azules y calientes.
- Los clusters con **mayor bp_rp** son más rojos y fríos.
- Los clusters con **GMAG más bajo** corresponden a estrellas más luminosas.
- Los clusters con **Teff alto** corresponden a estrellas tipo O/B/A.
- Los clusters con **Teff bajo** y **Lum-Flame bajo** corresponden a estrellas tipo K/M.

En particular, el perfil promedio deja ver que:

- El cluster 1 tiene **Teff muy alto** y GMAG negativo, consistente con estrellas azules y muy luminosas.
- El cluster 2 tiene **bp_rp muy alto** y temperaturas bajas, consistente con estrellas frías tipo M.
- El cluster 3 concentra valores de **GMAG alto** y **Teff bajo**, compatible con estrellas rojas menos luminosas.
- El cluster 0 vive en una zona intermedia de la secuencia principal.

### 5. Respuesta final

K-Means **no recupera OBAFGKM como categorías exactas**. Lo que hace es imponer una partición geométrica sobre el plano HR, y esa partición resulta parcialmente compatible con la física estelar.

La conclusión correcta no es que el algoritmo “clasificó estrellas”, sino que **descubrió regiones latentes interpretables** que se relacionan con temperatura, color y luminosidad. En otras palabras: el algoritmo encuentra estructura, pero el significado físico lo ponemos nosotros después.

### 6. Respuestas cortas a las preguntas de cierre

1. El K sugerido por silueta es **5**; K=7 difiere porque el criterio geométrico no coincide necesariamente con el número de clases humanas o físicas.
2. Los clusters son más bien **cortes sobre secuencias continuas**, no islas totalmente separadas.
3. Capturan zonas de la secuencia principal y regiones de estrellas azules/luminosas y rojas/frías.
4. K-Means **aproxima** OBAFGKM, pero no las recupera de forma limpia.
5. No es clasificación supervisada porque la etiqueta espectral **no se usó para entrenar**.
