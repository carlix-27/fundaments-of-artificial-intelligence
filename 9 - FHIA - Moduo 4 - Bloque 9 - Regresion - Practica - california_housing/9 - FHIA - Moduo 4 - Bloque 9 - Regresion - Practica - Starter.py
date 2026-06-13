# BLOQUE 9 - STARTER CODE
# Aprendizaje Supervisado: Regresión
# Mini Lab: Regresión Lineal como Diagnóstico de Modelos
#
# IMPORTANTE:
# El archivo california_housing.csv debe estar en la misma carpeta
# que este script o notebook.
#
# Objetivo:
# Implementar regresión lineal con ecuaciones normales y gradient descent,
# comparar resultados y diagnosticar generalización usando train/test.

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 0. SETUP — CARGA DEL DATASET LOCAL
# ============================================================

print("Cargando California Housing dataset desde CSV local...")

# TODO: asegurarse de que california_housing.csv esté en la misma carpeta
df = pd.read_csv("california_housing.csv")

target_col = "MedHouseVal"

X = df.drop(columns=[target_col]).values
y = df[target_col].values
feature_names = df.drop(columns=[target_col]).columns.tolist()

print(f"Shape X: {X.shape}")
print(f"Shape y: {y.shape}")
print(f"Features: {feature_names}")

print("\nPipeline ML: Datos -> Modelo -> Error -> Optimización -> Evaluación -> Diagnóstico")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalización: necesaria para Gradient Descent estable
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Agregar columna de unos para bias/intercepto w0
X_train_bias = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
X_test_bias = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]


# ============================================================
# 1. MÉTRICAS Y DIAGNÓSTICO
# ============================================================

def mse(y_true, y_pred):
    """Mean Squared Error."""
    # TODO: implementar MSE
    return None


def diagnosticar_modelo(train_mse, test_mse, threshold=0.15):
    """
    Diagnóstico simple basado en la diferencia entre error de train y test.

    Regla didáctica:
    - Train bajo + Test mucho más alto => posible overfitting
    - Ambos altos => posible underfitting
    - Ambos cercanos y bajos => buen ajuste relativo

    Nota: los umbrales son orientativos, no universales.
    """
    print("\nDiagnóstico del modelo:")

    gap = test_mse - train_mse
    print(f"Gap test-train: {gap}")

    # TODO: completar lógica de diagnóstico
    # Pista:
    # if gap > threshold:
    #     print("Posible OVERFITTING")
    # elif train_mse > 1.0 and test_mse > 1.0:
    #     print("Posible UNDERFITTING")
    # else:
    #     print("BUEN AJUSTE relativo")

    print("TODO: completar diagnóstico")


# ============================================================
# 2. BASELINE
# ============================================================

print("\n" + "="*60)
print("BASELINE: predecir siempre la media del training set")
print("="*60)

# TODO: predecir siempre la media de y_train
baseline_pred_train = None
baseline_pred_test = None

# TODO: calcular MSE de baseline
baseline_mse_train = None
baseline_mse_test = None

print(f"MSE Train baseline: {baseline_mse_train}")
print(f"MSE Test baseline:  {baseline_mse_test}")

print("\nInterpretación:")
print("Este baseline representa un modelo extremadamente simple.")
print("Cualquier modelo útil debería superar este punto de referencia.")


# ============================================================
# 3. ECUACIONES NORMALES
# ============================================================

def normal_equations(X, y):
    """
    Resolver regresión lineal con ecuaciones normales.

    Fórmula conceptual:
    w* = (X^T X)^(-1) X^T y

    Recomendación:
    usar np.linalg.pinv(X) @ y para mayor estabilidad numérica.
    """
    # TODO: implementar solución analítica
    w = None
    return w


print("\n" + "="*60)
print("MÉTODO 1: ECUACIONES NORMALES")
print("="*60)

start = time.time()
w_normal = normal_equations(X_train_bias, y_train)
time_normal = time.time() - start

# TODO: calcular predicciones
y_pred_train_normal = None
y_pred_test_normal = None

# TODO: calcular MSE
mse_train_normal = None
mse_test_normal = None

print(f"Tiempo entrenamiento: {time_normal:.4f} segundos")
print(f"MSE Train: {mse_train_normal}")
print(f"MSE Test:  {mse_test_normal}")

diagnosticar_modelo(mse_train_normal, mse_test_normal)


# ============================================================
# 4. GRADIENT DESCENT
# ============================================================

def gradient_descent(X, y, alpha=0.01, iterations=1000, verbose=True):
    """
    Implementar Batch Gradient Descent para regresión lineal.

    Pasos:
    1. Predecir: y_pred = X @ w
    2. Error: error = y_pred - y
    3. Gradiente MSE: gradient = (2/n) X^T (y_pred - y)
    4. Update: w = w - alpha * gradient
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    losses = []

    for i in range(iterations):
        # TODO 1: predicción
        y_pred = None

        # TODO 2: error
        error = None

        # TODO 3: gradiente
        gradient = None

        # TODO 4: update de pesos
        w = None

        # Monitorear pérdida
        loss = mse(y, y_pred)
        losses.append(loss)

        if verbose and (i % 100 == 0 or i == iterations - 1):
            print(f"Iteración {i}: MSE = {loss:.4f}")

    return w, losses


print("\n" + "="*60)
print("MÉTODO 2: GRADIENT DESCENT")
print("="*60)

start = time.time()
w_gd, losses = gradient_descent(
    X_train_bias,
    y_train,
    alpha=0.01,
    iterations=1000,
    verbose=True
)
time_gd = time.time() - start

# TODO: calcular predicciones
y_pred_train_gd = None
y_pred_test_gd = None

# TODO: calcular MSE
mse_train_gd = None
mse_test_gd = None

print(f"\nTiempo entrenamiento: {time_gd:.4f} segundos")
print(f"MSE Train: {mse_train_gd}")
print(f"MSE Test:  {mse_test_gd}")

diagnosticar_modelo(mse_train_gd, mse_test_gd)


# ============================================================
# 5. COMPARACIÓN
# ============================================================

print("\n" + "="*60)
print("COMPARACIÓN DE MÉTODOS")
print("="*60)

comparison = pd.DataFrame({
    "Método": ["Baseline", "Ecuaciones Normales", "Gradient Descent"],
    "MSE Train": [baseline_mse_train, mse_train_normal, mse_train_gd],
    "MSE Test": [baseline_mse_test, mse_test_normal, mse_test_gd],
    "Tiempo entrenamiento": ["-", time_normal, time_gd]
})

print(comparison)

print("\nPregunta de decisión:")
print("¿Cuál método elegirías y por qué?")
print("No respondas solo por el MSE: considerá tiempo, escalabilidad y estabilidad.")


# ============================================================
# 6. VISUALIZAR CONVERGENCIA
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel("Iteración")
plt.ylabel("MSE")
plt.title("Convergencia de Gradient Descent")
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 7. EXPERIMENTOS GUIADOS
# ============================================================

print("\n" + "="*60)
print("EXPERIMENTOS GUIADOS")
print("="*60)

print("""
1. Cambiá alpha a 0.001, 0.01 y 0.1.
   ¿Qué pasa con la convergencia?

2. Quitá la normalización.
   ¿Qué pasa con Gradient Descent?

3. Compará pesos de ecuaciones normales vs Gradient Descent.
   ¿Llegan al mismo resultado?

4. ¿Qué te dice la diferencia entre MSE train y MSE test?
""")
