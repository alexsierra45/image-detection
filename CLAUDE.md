# CLAUDE.md: image-detection

## Contexto
Proyecto de la asignatura de Visión por Computador (MUIA), trabajo "xview_detection": detectar 4 clases (Small car, Bus, Truck, Building) en tiles 640×640 de imágenes de satélite WorldView-3 (xView, 0,3 m/px). Es la continuación de `../image-recognition` y sigue **el mismo proceso** (bitácora, un factor por experimento, SLURM, entregas con informe y slides), pero esta carpeta es la raíz del proyecto: no depender de material de la carpeta hermana salvo para copiar infraestructura.

**Calendario de entregas** (DEC-004):
1. **Detectores de una etapa** (YOLO, SSD, RetinaNet…). Cierra el **2026-11-29 a las 23:00**. Fuentes del informe en `reports/delivery_01/`.
2. Detectores de dos etapas (Faster R-CNN, R-FCN…) comparados con los de la entrega 1 (*pendiente de enunciado*).
- Informe final y notebook de la mejor solución por fase: hasta el 2027-01-12.

**Challenge** (Codabench, competición 17966): un miembro del grupo sube `prediction.zip`, que contiene `prediction.json` en el formato de la celda 23 de [detector_example.ipynb](detector_example.ipynb). Métricas del leaderboard: **mAP@50, Mean Accuracy, Mean Precision, Mean Recall**, las mismas que la celda 19 del notebook del profesor (portadas en [det_metrics.py](det_metrics.py), CHG-003). **Decide mAP@50**. Las otras tres cuentan todas las detecciones exportadas, así que dependen del umbral `conf` de exportación (EXP-008).

## Entorno
- **Supercomputador Magerit (CeSViMa, UPM).** Las reglas generales están en `~/.claude/CLAUDE.md`. **Nada de entrenar ni evaluar en serio en el nodo de acceso** (`login*`: sin CUDA, 10 min de CPU por proceso). Todo run va por SLURM y **cada `sbatch` necesita la autorización del usuario**.
- **`.venv`:** módulo `Python/3.13.5-GCCcore-14.3.0` + `requirements.txt`: torch 2.6.0+cu124, torchvision 0.21.0, Ultralytics 8.4.218 (CHG-001, ISS-002).
  - Para recrearlo: `module load apps/2021 Python/3.13.5-GCCcore-14.3.0 && python -m venv .venv && TMPDIR=/home/y404/SCRATCH/y404034/tmp .venv/bin/pip install --no-compile -r requirements.txt && .venv/bin/pip cache purge`.
  - Pesos preentrenados, solo desde el nodo de acceso (descarga, sin cómputo): `source slurm/env.sh && .venv/bin/python tools/fetch_weights.py`.
- **Variables de entorno:** `slurm/env.sh` (cachés de Ultralytics y torch en `cache/`, `YOLO_OFFLINE=1`). Lo cargan los `.sbatch`.
- **Lanzar runs:**
  - `sbatch --job-name=EXP-0NN slurm/run.sbatch sweep.py expNNN`: standard-gpu, 1 L40S, 12 CPU, 48 GB, 3 h. Se ajusta con opciones de `sbatch` (`--time`, `--gres=gpu:a100:1`).
  - Submission: `sbatch --job-name=sub slurm/run.sbatch predict.py <run_id> [--conf …]`. Genera `submissions/<EXP>/prediction.zip`.
  - Notebook: `sbatch --job-name=EXP-0NN slurm/run.sbatch tools/run_notebook.py xview_detection.ipynb --python .venv/bin/python`.
  - Comprobar la GPU: `sbatch slurm/gpu_check.sbatch` (debug-gpu, 10 min).
  - Logs en `experiments/logs/<job>-<id>.out`. Seguimiento: `squeue -u $USER`, `sacct -j <id>` y `experiments/runs/<run_id>/results.csv` (YOLO) o `history.csv` (torchvision).
- **Dataset:** `xview_detection` es un enlace simbólico a `/home/y404/PROJECT/xview_detection` (solo lectura, no copiarlo). Contiene `xview_det_train.json` (cajas **xyxy absolutas sin recortar**, `category_id` = nombre de la clase), 7606 tiles de train y 852 de test (sin anotaciones).
- **Caché:** `cache` es un enlace simbólico a `/home/y404/SCRATCH/y404034/image-detection-cache`. Contiene `yolo/` (formato Ultralytics: symlinks a los tif + labels recortadas, CHG-002), `weights/`, `torch/` y `ultralytics/`. Se regenera con `python xview_det.py` (export) y `tools/fetch_weights.py`.
- **Split** (DEC-002): train/val agrupado por escena, fijo en `experiments/splits/val_scenes.json` (versionado). Todos los EXP usan el mismo.

## Código
- [xview_det.py](xview_det.py): anotaciones, split, recorte de cajas, export a YOLO, formato de submission (`to_submission`) y detecciones en npz.
- [det_metrics.py](det_metrics.py): métrica del challenge, en dos versiones:
  - la referencia literal del profesor;
  - una implementación rápida con registros por detección, que permite barrer umbrales y hacer bootstrap.
  - `python det_metrics.py` comprueba que coinciden.
- [train_det.py](train_det.py): `run(run_id, exp_id, phase, change, cfg)` entrena, predice val, evalúa y escribe:
  - `experiments/experiments.csv` (con lock);
  - `experiments/history/<run_id>.json`;
  - `experiments/preds/<run_id>_val.npz`.
- [tv_detectors.py](tv_detectors.py): SSD / SSDlite / RetinaNet de torchvision con el mismo protocolo.
- [sweep.py](sweep.py): definición de cada EXP (`RUNS`). Un factor por EXP, con la base indicada; se salta los runs ya registrados.
- [predict.py](predict.py): submission desde un run.
- `tools/compare.py`: bootstrap pareado por imagen de val entre dos runs.

## Regla principal: la bitácora
[LOGBOOK.md](LOGBOOK.md) es la memoria del proyecto y la fuente de verdad. El usuario quiere que no se pierda nada.

1. **Todo cambio** al notebook o al código se registra en LOGBOOK.md **dentro de la misma tarea**, como EXP, CHG, ISS, IDEA o DEC (ver las convenciones y plantillas del propio archivo). Para hacerlo, usar el skill `log-experiment`.
2. **Toda idea o cambio lleva fundamento teórico citado**: primero los documentos de `references/` (clave `[C#]` y página o sección), si no, papers o libros (`[R#]`, añadiéndolos a "Referencias externas"). Si algo es heurística empírica sin base teórica sólida, se dice explícitamente. No inventar citas: si no hay certeza de una referencia, buscarla o decirlo.
3. **Los resultados salen de evidencias**: `experiments/experiments.csv`, `experiments/history/`, `experiments/runs/<run_id>/` o los logs de SLURM. Sin ejecución, el estado es *pendiente*. Distinguir siempre código escrito / trabajo enviado (id) / trabajo terminado / resultado validado.
4. Un experimento cambia un solo factor respecto a su base, y su base se indica siempre. Con una sola semilla, diferencias de ≲ 1 pp de mAP@50 no son concluyentes: usar `tools/compare.py` (IC 95 % por bootstrap).
5. Mantener al día "Estado actual" y el "Índice de experimentos". Las incidencias y las ideas no se borran: cambian de estado (resuelta → CHG/EXP, probada → EXP).
6. Antes de terminar una tarea que tocó código, comprobar que la bitácora lo refleja.
7. Las incidencias detectadas al leer el código (bugs, inconsistencias) se registran como ISS aunque el usuario no las haya pedido.
8. **El test del challenge nunca se usa para decidir** (solo val). La nota de Codabench se anota en el EXP del envío.

Para proponer los siguientes pasos, usar el skill `next-ideas`.

## Notebook
- [xview_detection.ipynb](xview_detection.ipynb) es el entregable: la plantilla del profesor reescrita con el mejor modelo (DEC-005). [detector_example.ipynb](detector_example.ipynb) es el original y no se toca.
- El `EXP_ID` de su celda de tracking debe coincidir con la entrada de LOGBOOK.md.
- El usuario suele tener el notebook abierto en VS Code: al editarlo desde fuera, avisarle de que lo recargue.
- Se ejecuta sin interfaz con `tools/run_notebook.py` dentro de un trabajo de SLURM, nunca en el nodo de acceso.

## Entregas (informe + slides para Overleaf)
- Igual que en image-recognition: `make bundle D=NN` regenera macros, tablas y figuras desde `experiments/` (solo lectura), instala `drafts/` en `reports/overleaf/delivery_NN/sections/` sin pisar ediciones y genera el zip. Nueva entrega: comando `/new-delivery <n> <baseline>`.
- En los `.tex`, ninguna cifra a mano: `\res{clave}` y tablas de `generated/`. Lo que no se pueda rastrear va a `review_notes.md`.
- Figuras: estilo `plot_theme.py`, textos en inglés, PNG. Texto y pies, en español. Siempre: curvas de pérdida train/val, curvas PR por clase y matriz de confusión (con BACKGROUND) del modelo final.
- No hay LaTeX local: se compila en Overleaf (pdfLaTeX).

## Git
- Tras clonar: `bash tools/setup.sh` (activa `.githooks/pre-commit` y el diff de notebooks).
- El hook pre-commit bloquea commits con `.ipynb`/`.py` si `LOGBOOK.md` no va en el commit. No usar `--no-verify` salvo petición expresa del usuario.
- Los mensajes de commit citan los IDs de la bitácora, p. ej. `EXP-002: YOLO11s`. Solo se hace commit cuando el usuario lo pide.
- No versionar el dataset, los pesos (`*.pt`), las predicciones (`experiments/preds/`, `submissions/`) ni los PDF de `references/`.
