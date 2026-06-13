# Práctica Bloque 14 · Arquitecturas Modernas e IA Híbrida

## Contenido del pack

- `Bloque 14 - Practica Hogarena - Guia Estudiante.docx`: guía completa para estudiantes.
- `Bloque 14 - Plantilla Informe.docx`: plantilla para entregar el informe.
- `Bloque 14 - Practica Hogarena.py`: script ejecutable de la práctica.

## Instalación

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

Si TensorFlow no corre localmente, usar Google Colab.

## Ejecución recomendada

```bash
python "Bloque 14 - Practica Hogarena.py" --quick
```

## Ejecución completa

```bash
python "Bloque 14 - Practica Hogarena.py" --full
```

## Salida esperada

El script crea la carpeta `bloque14_outputs` con gráficos, métricas y una plantilla de respuestas en Markdown:

- `A_fashion_mnist_samples.png`
- `A_mlp_vs_cnn_accuracy.png`
- `A_mlp_vs_cnn_loss.png`
- `B_time_series_sample.png`
- `B_predictions_seq20.png`, `B_predictions_seq50.png`
- `B_training_seq20.png`, `B_training_seq50.png`
- `C_attention_matrix_toy.png`
- `resultados_bloque14.json`
- `plantilla_respuestas_bloque14.md`

## Idea central

La práctica no evalúa solo si el código corre. Evalúa si el estudiante puede explicar por qué una arquitectura funciona mejor cuando su sesgo inductivo coincide con la estructura del dato, y cómo los sistemas modernos combinan modelos, recuperación, herramientas, memoria y control.

Usá --quick como modo recomendado para completar la práctica. Usá --full solo si tenés tiempo y recursos suficientes; puede mejorar la estabilidad de los resultados, pero no es necesario para aprobar la actividad.