"""
Bloque 11 - Aprendizaje No Supervisado
Práctica: Descubrimiento de regiones latentes en el Diagrama HR con Gaia

El código ya está provisto. La tarea del estudiante es ejecutar, observar,
interpretar y criticar los resultados.

Dataset esperado:
    gaia_stars_sample_14k.csv

Columnas mínimas esperadas:
    BPmag, RPmag, GMAG, spectral_class
Columnas útiles para interpretación:
    Teff, Rad, Lum-Flame, Plx, Dist

Idea central:
    No entrenamos con la clase espectral. Usamos solo el plano HR:
        bp_rp = BPmag - RPmag
        GMAG  = magnitud absoluta en banda G
    Luego comparamos los clusters con OBAFGKM recién al final.
"""

from __future__ import annotations

import os
from pathlib import Path
import warnings

# Evitar problemas de performance por múltiples threads en algunas instalaciones
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# CONFIGURACIÓN
# =========================
DATASET = "gaia_stars_sample_14k.csv"
OUTPUT_DIR = Path("bloque11_gaia_hr_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
K_RANGE = range(2, 11)
SILHOUETTE_SAMPLE_SIZE = 2000
K_DOMAIN = 7  # O, B, A, F, G, K, M


def purity_score(y_true: pd.Series, labels: np.ndarray) -> float:
    """Pureza posterior: NO es accuracy supervisada; solo mide alineación cluster-clase al final."""
    table = pd.crosstab(labels, y_true)
    return table.max(axis=1).sum() / table.values.sum()


def savefig(name: str) -> None:
    path = OUTPUT_DIR / name
    plt.savefig(path, dpi=160, bbox_inches="tight")
    print(f"✓ Gráfico guardado: {path}")


def make_hr_plot(df: pd.DataFrame, color_values, title: str, filename: str, colorbar_label: str = "Cluster"):
    plt.figure(figsize=(9, 7))
    sc = plt.scatter(
        df["bp_rp"], df["GMAG"],
        c=color_values,
        s=7,
        alpha=0.42,
        edgecolors="none",
        rasterized=True,
    )
    plt.gca().invert_yaxis()  # En magnitudes, menor valor = más luminoso
    plt.xlabel("BP - RP (color)")
    plt.ylabel("Magnitud absoluta G (GMAG)")
    plt.title(title)
    plt.grid(alpha=0.25)
    cbar = plt.colorbar(sc)
    cbar.set_label(colorbar_label)
    savefig(filename)
    plt.close()


def main() -> None:
    print("=" * 78)
    print("BLOQUE 11 - PRÁCTICA GAIA / DIAGRAMA HR")
    print("Descubrimiento de regiones latentes con K-Means")
    print("=" * 78)
    print("\nEl código ya está provisto. La tarea es interpretar los resultados.\n")

    dataset_path = Path(DATASET)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No se encontró {DATASET}. Colocá el CSV en la misma carpeta que este script."
        )

    # =========================
    # A. CARGA Y PREPARACIÓN
    # =========================
    df = pd.read_csv(dataset_path)
    print(f"✓ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Columnas: {list(df.columns)}\n")

    required = {"BPmag", "RPmag", "GMAG", "spectral_class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    if "bp_rp" not in df.columns:
        df["bp_rp"] = df["BPmag"] - df["RPmag"]

    # Nos quedamos con filas válidas para HR
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["bp_rp", "GMAG", "spectral_class"]).copy()
    print(f"✓ Filas válidas para el análisis HR: {len(df)}")

    print("\nDistribución de clases espectrales en el dataset balanceado:")
    print(df["spectral_class"].value_counts().sort_index().to_string())
    print("\nIMPORTANTE: spectral_class NO se usa para entrenar K-Means. Se revela al final.\n")

    # =========================
    # B. FEATURES HR Y ESCALADO
    # =========================
    hr_features = ["bp_rp", "GMAG"]
    X = df[hr_features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Features usadas por K-Means:")
    print("  bp_rp = BPmag - RPmag  (color)")
    print("  GMAG  = magnitud absoluta en banda G")
    print("✓ Estandarización aplicada antes de K-Means.\n")

    # =========================
    # C. K-MEANS: CODO + SILUETA
    # =========================
    inertias = []
    silhouettes = []
    purities = []
    labels_by_k = {}

    print("Evaluando K-Means para K=2..10")
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE, algorithm="lloyd")
        labels = km.fit_predict(X_scaled)
        labels_by_k[k] = labels
        inertias.append(km.inertia_)
        sil = silhouette_score(
            X_scaled,
            labels,
            sample_size=min(SILHOUETTE_SAMPLE_SIZE, len(df)),
            random_state=RANDOM_STATE,
        )
        silhouettes.append(sil)
        pur = purity_score(df["spectral_class"], labels)
        purities.append(pur)
        print(f"K={k:2d} | Inercia={km.inertia_:9.2f} | Silueta={sil:.3f} | Pureza posterior={pur:.3f}")

    k_silhouette = list(K_RANGE)[int(np.argmax(silhouettes))]
    print("\nK sugerido por máxima silueta:", k_silhouette)
    print("K de comparación por dominio espectral:", K_DOMAIN, "(O, B, A, F, G, K, M)")
    print("\nLa pureza posterior se calcula después de revelar spectral_class; no es accuracy supervisada.\n")

    # Gráfico codo/silueta/pureza
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    ks = list(K_RANGE)
    axes[0].plot(ks, inertias, marker="o", linewidth=2)
    axes[0].set_title("Método del codo")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inercia / WCSS")
    axes[0].grid(alpha=0.3)

    axes[1].plot(ks, silhouettes, marker="s", linewidth=2)
    axes[1].axvline(k_silhouette, linestyle="--", alpha=0.7)
    axes[1].set_title("Coeficiente de silueta")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silueta")
    axes[1].grid(alpha=0.3)

    axes[2].plot(ks, purities, marker="^", linewidth=2)
    axes[2].axvline(K_DOMAIN, linestyle="--", alpha=0.7)
    axes[2].set_title("Pureza posterior vs clase espectral")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Pureza")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Selección de K: criterio geométrico vs comparación posterior", fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig("01_k_selection_elbow_silhouette_purity.png")
    plt.close()

    # =========================
    # D. HR DIAGRAM: CLUSTERS
    # =========================
    labels_sil = labels_by_k[k_silhouette]
    labels_k7 = labels_by_k[K_DOMAIN]
    df["cluster_silhouette"] = labels_sil
    df["cluster_k7"] = labels_k7

    make_hr_plot(
        df,
        labels_sil,
        f"Diagrama HR coloreado por clusters K-Means (K={k_silhouette}, sugerido por silueta)",
        "02_hr_clusters_k_silhouette.png",
        "Cluster",
    )

    make_hr_plot(
        df,
        labels_k7,
        "Diagrama HR coloreado por clusters K-Means (K=7, comparación con OBAFGKM)",
        "03_hr_clusters_k7.png",
        "Cluster",
    )

    # Diagrama HR por clase espectral: usar códigos numéricos para colorbar simple
    class_order = ["O", "B", "A", "F", "G", "K", "M"]
    class_to_num = {c: i for i, c in enumerate(class_order)}
    y_num = df["spectral_class"].map(class_to_num)

    plt.figure(figsize=(9, 7))
    sc = plt.scatter(
        df["bp_rp"], df["GMAG"],
        c=y_num,
        s=7,
        alpha=0.42,
        edgecolors="none",
        rasterized=True,
    )
    plt.gca().invert_yaxis()
    plt.xlabel("BP - RP (color)")
    plt.ylabel("Magnitud absoluta G (GMAG)")
    plt.title("Diagrama HR coloreado por clase espectral real (revelada al final)")
    plt.grid(alpha=0.25)
    cbar = plt.colorbar(sc, ticks=list(range(len(class_order))))
    cbar.ax.set_yticklabels(class_order)
    cbar.set_label("Clase espectral")
    savefig("04_hr_spectral_class_real.png")
    plt.close()

    # =========================
    # E. MATRICES CLUSTER x CLASE
    # =========================
    for label_col, name in [("cluster_silhouette", f"K={k_silhouette}"), ("cluster_k7", "K=7")]:
        tab_counts = pd.crosstab(df[label_col], df["spectral_class"])
        tab_norm = pd.crosstab(df[label_col], df["spectral_class"], normalize="index")
        tab_counts.to_csv(OUTPUT_DIR / f"cluster_vs_spectral_class_counts_{label_col}.csv")
        tab_norm.to_csv(OUTPUT_DIR / f"cluster_vs_spectral_class_normalized_{label_col}.csv")

        print("\n" + "=" * 78)
        print(f"Tabla cluster × clase espectral ({name}) - conteos")
        print("=" * 78)
        print(tab_counts.to_string())
        print("\nTabla normalizada por cluster")
        print(tab_norm.round(3).to_string())
        print(f"Pureza posterior ({name}): {purity_score(df['spectral_class'], df[label_col]):.3f}")

    # =========================
    # F. PERFIL FÍSICO DE CLUSTERS
    # =========================
    interpret_cols = [c for c in ["bp_rp", "GMAG", "Teff", "Rad", "Lum-Flame", "Plx", "Dist"] if c in df.columns]
    profiles = df.groupby("cluster_k7")[interpret_cols].agg(["mean", "median", "std", "count"])
    profiles.to_csv(OUTPUT_DIR / "cluster_profiles_k7.csv")

    print("\n" + "=" * 78)
    print("Perfil físico resumido por cluster (K=7)")
    print("=" * 78)
    print(df.groupby("cluster_k7")[interpret_cols].mean().round(3).to_string())

    # Gráfico de medias principales por cluster K=7
    main_profile_cols = [c for c in ["bp_rp", "GMAG", "Teff", "Lum-Flame"] if c in df.columns]
    fig, axes = plt.subplots(1, len(main_profile_cols), figsize=(5 * len(main_profile_cols), 4))
    if len(main_profile_cols) == 1:
        axes = [axes]
    means = df.groupby("cluster_k7")[main_profile_cols].mean()
    for ax, col in zip(axes, main_profile_cols):
        ax.bar(means.index.astype(str), means[col])
        ax.set_title(f"Media de {col}")
        ax.set_xlabel("Cluster K=7")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Perfil físico promedio de los clusters (K=7)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig("05_cluster_profiles_k7.png")
    plt.close()

    # =========================
    # G. PCA COMPLEMENTARIO
    # =========================
    # PCA sobre HR features. Como son 2 variables, PCA solo rota/reescala el mismo espacio.
    # Se incluye para conectar con la clase y mostrar varianza explicada.
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print("\nPCA sobre [bp_rp, GMAG]")
    print(f"PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print("Nota: con dos variables, PCA no agrega información; solo cambia coordenadas.")

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_k7, s=7, alpha=0.42, edgecolors="none", rasterized=True)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title("PCA del espacio HR coloreado por clusters K=7")
    plt.grid(alpha=0.25)
    plt.colorbar(sc, label="Cluster K=7")
    savefig("06_pca_hr_features_k7.png")
    plt.close()

    # Guardar dataset con clusters
    out_csv = OUTPUT_DIR / "gaia_stars_sample_14k_with_clusters.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ Dataset con clusters guardado: {out_csv}")

    # =========================
    # CIERRE
    # =========================
    print("\n" + "=" * 78)
    print("PREGUNTAS DE CIERRE PARA EL INFORME")
    print("=" * 78)
    print("1. ¿El K sugerido por silueta coincide con K=7? ¿Por qué podría diferir?")
    print("2. ¿Los clusters aparecen como islas separadas o como cortes sobre secuencias continuas?")
    print("3. ¿Qué regiones del diagrama HR parecen capturar los clusters?")
    print("4. ¿K-Means recupera OBAFGKM o impone una partición geométrica diferente?")
    print("5. ¿Por qué este resultado NO debe interpretarse como clasificación supervisada?")
    print("=" * 78)
    print(f"\n✓ Práctica finalizada. Revisá la carpeta: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
