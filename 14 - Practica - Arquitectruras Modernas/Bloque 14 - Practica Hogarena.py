"""
BLOQUE 14 - PRÁCTICA HOGAREÑA
Arquitecturas Modernas, Atención e IA Híbrida
Universidad Austral - Fundamentos e Historia de la IA

Objetivo:
1. Comparar MLP vs CNN para imágenes.
2. Comparar SimpleRNN vs LSTM para series temporales.
3. Observar un cálculo toy de self-attention.
4. Generar resultados para responder preguntas conceptuales.
5. Analizar RAG e IA híbrida.
6. Distinguir objetivos discriminativos y generativos.

Uso sugerido:
    python "Bloque 14 - Practica Hogarena.py" --quick
    python "Bloque 14 - Practica Hogarena.py" --full

Notas:
- Los números exactos pueden variar según versión de TensorFlow, hardware y semilla.
- El foco de la práctica es interpretar tendencias, no obtener un benchmark perfecto.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as exc:
    raise RuntimeError(
        "No se pudo importar TensorFlow. Instalá dependencias con: "
        "pip install tensorflow numpy matplotlib scikit-learn"
    ) from exc


CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Práctica Bloque 14 - Arquitecturas Modernas")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Modo rápido: menos datos y epochs.")
    mode.add_argument("--full", action="store_true", help="Modo completo: más datos y epochs.")
    parser.add_argument("--out", type=str, default="bloque14_outputs", help="Carpeta de salida.")
    parser.add_argument("--skip-pixel-shuffle", action="store_true", help="No ejecutar CNN con píxeles mezclados.")
    parser.add_argument("--skip-seq100", action="store_true", help="No ejecutar experimento opcional seq_length=100.")
    return parser.parse_args()


def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_out(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_mlp(input_dim=784):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax"),
    ], name="MLP")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax"),
    ], name="CNN")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_fashion_samples(x_train, y_train, out):
    plt.figure(figsize=(12, 4))
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.imshow(x_train[i], cmap="gray")
        plt.title(CLASS_NAMES[int(y_train[i])], fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out / "A_fashion_mnist_samples.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_histories(histories, metric, title, out_file):
    plt.figure(figsize=(10, 5))
    for label, hist in histories.items():
        plt.plot(hist.history[metric], label=f"{label} train", linewidth=2)
        val_key = "val_" + metric
        if val_key in hist.history:
            plt.plot(hist.history[val_key], label=f"{label} val", linestyle="--", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()


def run_image_experiment(out: Path, quick: bool, skip_pixel_shuffle: bool):
    print("\n" + "=" * 80)
    print("PARTE A - MLP vs CNN para imágenes")
    print("=" * 80)

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    if quick:
        train_n, test_n, epochs = 12000, 3000, 4
    else:
        train_n, test_n, epochs = 60000, 10000, 10

    x_train = x_train[:train_n]
    y_train = y_train[:train_n]
    x_test = x_test[:test_n]
    y_test = y_test[:test_n]

    plot_fashion_samples(x_train, y_train, out)

    x_train_flat = x_train.reshape(-1, 784)
    x_test_flat = x_test.reshape(-1, 784)
    x_train_cnn = x_train[..., np.newaxis]
    x_test_cnn = x_test[..., np.newaxis]

    results = {}
    histories = {}

    print("\nEntrenando MLP...")
    mlp = build_mlp()
    start = time.time()
    histories["MLP"] = mlp.fit(
        x_train_flat, y_train,
        epochs=epochs,
        batch_size=128,
        validation_split=0.1,
        verbose=1,
    )
    train_time = time.time() - start
    loss, acc = mlp.evaluate(x_test_flat, y_test, verbose=0)
    results["MLP"] = {
        "params": int(mlp.count_params()),
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "train_time_sec": float(train_time),
    }

    print("\nEntrenando CNN...")
    cnn = build_cnn()
    start = time.time()
    histories["CNN"] = cnn.fit(
        x_train_cnn, y_train,
        epochs=epochs,
        batch_size=128,
        validation_split=0.1,
        verbose=1,
    )
    train_time = time.time() - start
    loss, acc = cnn.evaluate(x_test_cnn, y_test, verbose=0)
    results["CNN"] = {
        "params": int(cnn.count_params()),
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "train_time_sec": float(train_time),
    }

    if not skip_pixel_shuffle:
        print("\nEntrenando CNN con estructura espacial destruida (píxeles mezclados)...")
        rng = np.random.default_rng(42)
        perm = rng.permutation(784)
        x_train_shuffled = x_train.reshape(-1, 784)[:, perm].reshape(-1, 28, 28, 1)
        x_test_shuffled = x_test.reshape(-1, 784)[:, perm].reshape(-1, 28, 28, 1)
        cnn_shuffled = build_cnn()
        start = time.time()
        histories["CNN pixel shuffle"] = cnn_shuffled.fit(
            x_train_shuffled, y_train,
            epochs=epochs,
            batch_size=128,
            validation_split=0.1,
            verbose=1,
        )
        train_time = time.time() - start
        loss, acc = cnn_shuffled.evaluate(x_test_shuffled, y_test, verbose=0)
        results["CNN_pixel_shuffle"] = {
            "params": int(cnn_shuffled.count_params()),
            "test_loss": float(loss),
            "test_accuracy": float(acc),
            "train_time_sec": float(train_time),
        }

    plot_histories(histories, "accuracy", "Parte A - Accuracy: train vs validation", out / "A_mlp_vs_cnn_accuracy.png")
    plot_histories(histories, "loss", "Parte A - Loss: train vs validation", out / "A_mlp_vs_cnn_loss.png")

    print("\nResumen Parte A")
    print(f"{'Modelo':<22} {'Parámetros':>12} {'Acc':>10} {'Loss':>10} {'Tiempo(s)':>10}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<22} {r['params']:>12,} {r['test_accuracy']:>10.4f} {r['test_loss']:>10.4f} {r['train_time_sec']:>10.1f}")

    return results


def generate_series(n_steps=1200, noise=0.10):
    t = np.linspace(0, 120, n_steps)
    series = np.sin(0.1 * t) + 0.5 * np.sin(0.3 * t) + 0.25 * np.sin(0.03 * t) + noise * np.random.randn(n_steps)
    return t, series.astype("float32")


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    X = np.array(X).reshape(-1, seq_length, 1)
    y = np.array(y)
    return X, y


def build_rnn(seq_length):
    model = keras.Sequential([
        layers.Input(shape=(seq_length, 1)),
        layers.SimpleRNN(32, activation="tanh"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ], name=f"SimpleRNN_seq{seq_length}")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_lstm(seq_length):
    model = keras.Sequential([
        layers.Input(shape=(seq_length, 1)),
        layers.LSTM(32, activation="tanh"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ], name=f"LSTM_seq{seq_length}")
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def plot_series(t, series, out):
    plt.figure(figsize=(12, 4))
    plt.plot(t[:400], series[:400], linewidth=1.5)
    plt.xlabel("t")
    plt.ylabel("valor")
    plt.title("Parte B - Serie temporal sintética")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "B_time_series_sample.png", dpi=150, bbox_inches="tight")
    plt.close()


def run_single_sequence_experiment(series, seq_length, out, quick):
    X, y = create_sequences(series, seq_length)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    epochs = 5 if quick else 15
    batch_size = 32

    results = {}
    histories = {}
    preds = {}

    for label, builder in [("RNN", build_rnn), ("LSTM", build_lstm)]:
        print(f"\nEntrenando {label} con seq_length={seq_length}...")
        model = builder(seq_length)
        hist = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1,
        )
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        y_pred = model.predict(X_test, verbose=0).flatten()
        histories[label] = hist
        preds[label] = y_pred
        results[label] = {"test_mse": float(loss), "test_mae": float(mae), "params": int(model.count_params())}

    # Plot predictions
    plot_range = min(180, len(y_test))
    plt.figure(figsize=(12, 5))
    plt.plot(y_test[:plot_range], label="Real", linewidth=2, alpha=0.75)
    plt.plot(preds["RNN"][:plot_range], label="RNN", linewidth=1.5, alpha=0.8)
    plt.plot(preds["LSTM"][:plot_range], label="LSTM", linewidth=1.5, alpha=0.8)
    plt.xlabel("paso de test")
    plt.ylabel("valor")
    plt.title(f"Parte B - Predicciones RNN vs LSTM (seq_length={seq_length})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / f"B_predictions_seq{seq_length}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot validation loss
    plt.figure(figsize=(10, 5))
    for label, hist in histories.items():
        plt.plot(hist.history["val_loss"], label=f"{label} val", linewidth=2)
        plt.plot(hist.history["loss"], label=f"{label} train", linestyle="--", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"Parte B - Entrenamiento RNN vs LSTM (seq_length={seq_length})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / f"B_training_seq{seq_length}.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results


def run_sequence_experiment(out: Path, quick: bool, skip_seq100: bool):
    print("\n" + "=" * 80)
    print("PARTE B - RNN vs LSTM para series temporales")
    print("=" * 80)
    t, series = generate_series(n_steps=1000 if quick else 1600)
    plot_series(t, series, out)

    seq_lengths = [20, 50]
    if not quick and not skip_seq100:
        seq_lengths.append(100)

    all_results = {}
    for seq_length in seq_lengths:
        all_results[f"seq_{seq_length}"] = run_single_sequence_experiment(series, seq_length, out, quick)

    print("\nResumen Parte B")
    print(f"{'Seq':<8} {'Modelo':<10} {'Parámetros':>12} {'MSE':>12} {'MAE':>12}")
    print("-" * 62)
    for seq, res in all_results.items():
        for model, r in res.items():
            print(f"{seq:<8} {model:<10} {r['params']:>12,} {r['test_mse']:>12.6f} {r['test_mae']:>12.6f}")

    return all_results


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def run_attention_toy(out: Path):
    print("\n" + "=" * 80)
    print("PARTE C - Toy de self-attention")
    print("=" * 80)
    rng = np.random.default_rng(7)
    n_tokens = 5
    d_model = 4
    d_k = 3
    d_v = 3

    tokens = ["El", "gato", "vio", "al", "perro"]
    X = rng.normal(size=(n_tokens, d_model))
    W_Q = rng.normal(size=(d_model, d_k))
    W_K = rng.normal(size=(d_model, d_k))
    W_V = rng.normal(size=(d_model, d_v))

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    S = (Q @ K.T) / np.sqrt(d_k)
    A = softmax(S, axis=1)
    Z = A @ V

    print("Formas:")
    print("X:", X.shape, "Q:", Q.shape, "K:", K.shape, "V:", V.shape)
    print("S = QK^T/sqrt(d_k):", S.shape)
    print("A = softmax(S):", A.shape)
    print("Z = AV:", Z.shape)
    print("\nCada fila de A suma:", np.round(A.sum(axis=1), 6))
    print("\nMatriz de atención A:")
    print(np.round(A, 3))

    plt.figure(figsize=(7, 6))
    plt.imshow(A, aspect="auto")
    plt.colorbar(label="peso de atención")
    plt.xticks(range(n_tokens), tokens)
    plt.yticks(range(n_tokens), tokens)
    plt.xlabel("Keys: tokens atendidos")
    plt.ylabel("Queries: token que atiende")
    plt.title("Parte C - Matriz de atención toy")
    for i in range(n_tokens):
        for j in range(n_tokens):
            plt.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", color="white" if A[i, j] > 0.45 else "black", fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "C_attention_matrix_toy.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "tokens": tokens,
        "shapes": {"X": X.shape, "Q": Q.shape, "K": K.shape, "V": V.shape, "S": S.shape, "A": A.shape, "Z": Z.shape},
        "attention_matrix": A.tolist(),
        "row_sums": A.sum(axis=1).tolist(),
    }



def write_answers_template(out: Path):
    template = """# Respuestas conceptuales · Práctica Bloque 14

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
"""
    (out / "plantilla_respuestas_bloque14.md").write_text(template, encoding="utf-8")


def save_results(results, out: Path):
    serializable = {}
    for k, v in results.items():
        try:
            json.dumps(v)
            serializable[k] = v
        except TypeError:
            serializable[k] = str(v)
    with open(out / "resultados_bloque14.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    quick = True if args.quick or not args.full else False
    set_seed(42)
    out = ensure_out(args.out)

    print("TensorFlow version:", tf.__version__)
    print("GPU disponible:", tf.config.list_physical_devices("GPU"))
    print("Modo:", "quick" if quick else "full")
    print("Carpeta de salida:", out.resolve())

    results = {}
    results["parte_A_imagenes"] = run_image_experiment(out, quick, args.skip_pixel_shuffle)
    results["parte_B_series"] = run_sequence_experiment(out, quick, args.skip_seq100)
    results["parte_C_attention_toy"] = run_attention_toy(out)
    save_results(results, out)
    write_answers_template(out)

    print("\n" + "=" * 80)
    print("PRÁCTICA COMPLETADA")
    print("=" * 80)
    print("Archivos generados en:", out.resolve())
    print("- A_fashion_mnist_samples.png")
    print("- A_mlp_vs_cnn_accuracy.png")
    print("- A_mlp_vs_cnn_loss.png")
    print("- B_time_series_sample.png")
    print("- B_predictions_seq*.png")
    print("- B_training_seq*.png")
    print("- C_attention_matrix_toy.png")
    print("- resultados_bloque14.json")
    print("- plantilla_respuestas_bloque14.md")
    print("\nRecordá: el entregable importante no son solo los números, sino la interpretación arquitectural.")


if __name__ == "__main__":
    main()
