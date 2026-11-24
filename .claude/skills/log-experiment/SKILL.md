---
name: log-experiment
description: Registra en LOGBOOK.md un experimento, cambio, incidencia, idea o decisión del proyecto xview_detection, con fundamento teórico citado y resultados sacados de experiments/. Usar tras modificar el notebook o el código, tras un entrenamiento (trabajo de SLURM), al detectar un problema y antes de cualquier commit.
argument-hint: "[qué se hizo, opcional]"
---

# Registrar en la bitácora

Argumento del usuario (puede estar vacío): $ARGUMENTS

## 1. Reunir la evidencia (no preguntar lo que se puede leer)
- `git status` y `git diff -- '*.ipynb' '*.py'`. Gracias al textconv, el diff de los notebooks muestra solo el código de las celdas. Si aún no hay commits, leer el notebook directamente.
- `experiments/experiments.csv`: filas nuevas desde la última entrada de la bitácora. `id` es la clave del run.
- Por cada run nuevo:
  - `experiments/history/<id>.json`: configuración real, métricas del challenge (mAP@50, mean accuracy/precision/recall), AP/recall/precisión por clase, matriz de confusión con BACKGROUND, info de SLURM.
  - Curvas: `experiments/runs/<id>/results.csv` (YOLO: pérdidas box/cls/dfl de train y val, mAP por época) o `history.csv` (torchvision: pérdidas y mAP@50 de val). ¿Underfitting, overfitting, divergencia? ¿En qué época está el mejor modelo? ¿Hacía falta entrenar más?
  - Log de SLURM `experiments/logs/<job>-<jobid>.out`: avisos (p. ej. `NMS time limit exceeded`, que trunca detecciones), errores, tiempo por época.
  - Comparación con la base: `python tools/compare.py <base> <id>` (bootstrap pareado, IC 95 %).
- Estado de los trabajos: `squeue -u $USER`, `sacct -j <jobid>`. Distinguir enviado / terminado / validado.
- LOGBOOK.md: último ID de cada tipo, estado actual, ideas e incidencias abiertas.

## 2. Clasificar
- **EXP**: cambia el modelo, los datos, el preprocesado o el entrenamiento. Tiene, o tendrá, métricas.
- **CHG**: cambio de código sin efecto en el modelo (refactor, instrumentación, utilidades).
- **ISS**: bug, inconsistencia o problema del entorno.
- **IDEA**: mejora todavía no probada.
- **DEC**: elección entre alternativas.

Un mismo cambio puede generar varias entradas: por ejemplo, un EXP que resuelve una ISS y cierra una IDEA.

## 3. Fundamentar
- Buscar primero en `references/` (índice en `references/README.md`): leer la parte relevante del PDF y citarla como `[C#, p. X]`.
- Si no está ahí, usar la bibliografía canónica (`[R#]`, añadiéndola a "Referencias externas" si es nueva). Verificar la referencia si hay alguna duda; no inventar citas.
- Si la base es solo empírica o heurística, decirlo.
- Preferir argumentos cuantitativos con los datos reales: número de parámetros, frecuencias de clase, varianza de las estimaciones, escalas de las activaciones.

## 4. Preguntar solo lo imprescindible
La hipótesis o motivación del usuario, si no es obvia por el código. Nunca rellenar resultados de memoria: si un run no se ha ejecutado, la entrada queda *pendiente*.

## 5. Escribir
- Usar las plantillas que hay al final de LOGBOOK.md. Para un EXP, indicar la base y el único factor que cambia. Comparar con la base usando las mismas métricas (mAP@50 decide; además mRecall, mPrecision, mAcc, AP por clase) y el IC del bootstrap.
- Actualizar "Estado actual" (el mejor experimento según mAP@50 de val) y el "Índice de experimentos".
- Actualizar el estado de las ISS y las IDEA afectadas (resuelta → CHG/EXP, probada → EXP), sin borrarlas.
- Si el `EXP_ID` del notebook no coincide con la entrada, avisar al usuario.
- Nunca usar la nota de Codabench para decidir: se anota en el EXP del envío.

## 6. Cerrar
Resumir al usuario en 2 o 3 líneas qué IDs se crearon o actualizaron y cuál es el siguiente paso recomendado.
