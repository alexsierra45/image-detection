# image-detection: xView object detection

Trabajo de Visión por Computador (MUIA, UPM, curso 2026-27): detectar **Small car, Bus, Truck y Building** en tiles 640×640 de imágenes de satélite WorldView-3 del dataset [xView](http://xviewdataset.org/) (0,3 m/píxel).

- **Datos:** 7606 tiles de train con anotaciones y 852 de test sin ellas.
- **Evaluación:** se sube a [Codabench](https://www.codabench.org/competitions/17966/) con cuatro métricas: **mAP@50** (la principal), Mean Accuracy, Mean Precision y Mean Recall.

| Entrega | Contenido | Fecha límite |
|---|---|---|
| 1 | Detectores de una etapa: YOLO, SSD, RetinaNet | 2026-11-29 |
| 2 | Detectores de dos etapas (Faster R-CNN…) comparados con los de la entrega 1 | por anunciar |
| Final | Informe y notebook de la mejor solución | 2027-01-12 |

Todo corre en el supercomputador **Magerit** (CeSViMa) mediante SLURM. La memoria del proyecto está en [LOGBOOK.md](LOGBOOK.md): cada experimento, cambio, incidencia y decisión, con su justificación y sus resultados. Las convenciones de trabajo están en [CLAUDE.md](CLAUDE.md).

## Enfoque

- **Modelos:**
  - YOLO11 en tallas n, s y m, con [Ultralytics](https://github.com/ultralytics/ultralytics).
  - SSD300-VGG16 y RetinaNet-R50-FPN v2, de torchvision.
  - Todos parten de pesos COCO, salvo el experimento entrenado desde cero.
- **Mismo protocolo para todos los modelos:**
  - Validación con el 10 % de las **escenas** (no de los tiles), para no mezclar en train y val tiles vecinos de la misma escena. El split es fijo y está en `experiments/splits/val_scenes.json`.
  - Métricas calculadas con una copia exacta del código de evaluación del profesor ([det_metrics.py](det_metrics.py)), comprobada contra el original.
  - Inferencia común: `conf` 0,001, `max_det` 1500 (hay imágenes con hasta 1029 objetos) y NMS IoU 0,7.
- **Experimentos:** cada uno cambia un solo factor respecto a su base, y el test de Codabench nunca se usa para decidir. El índice y su estado están en [LOGBOOK.md](LOGBOOK.md#índice-de-experimentos).

## Estructura

```
xview_det.py         datos: anotaciones, split por escena, export a YOLO, formato de submission
det_metrics.py       métricas del challenge (referencia del profesor + versión rápida con bootstrap)
train_det.py         un run: entrena → predice val → evalúa → experiments.csv
tv_detectors.py      SSD / RetinaNet de torchvision
sweep.py             definición de los experimentos (python sweep.py exp001)
predict.py           genera submissions/<EXP>/prediction.zip a partir de un run
tools/               gpu_check, fetch_weights, compare (bootstrap), infer_sweep (umbral de confianza), run_notebook
slurm/               run.sbatch (lanzador genérico), gpu_check.sbatch, env.sh
experiments/         experiments.csv, history/, runs/, logs/, splits/   (preds/ no se versiona)
detector_example.ipynb   notebook de ejemplo del profesor (sin modificar)
LOGBOOK.md           bitácora del proyecto
```

`xview_detection` y `cache` son enlaces simbólicos que no se versionan:
- `xview_detection` → `/home/y404/PROJECT/xview_detection`, el dataset (solo lectura).
- `cache` → `/home/y404/SCRATCH/y404034/image-detection-cache`, con el export YOLO y los pesos preentrenados.

## Puesta en marcha (Magerit)

```bash
module load apps/2021 Python/3.13.5-GCCcore-14.3.0
python -m venv .venv
TMPDIR=/home/y404/SCRATCH/y404034/tmp .venv/bin/pip install --no-compile -r requirements.txt
.venv/bin/pip cache purge                    # la caché de pip ocupa varios GB de la cuota del grupo
bash tools/setup.sh                          # hook pre-commit y diff de notebooks

ln -s /home/y404/PROJECT/xview_detection xview_detection
ln -s /home/y404/SCRATCH/y404034/image-detection-cache cache
source slurm/env.sh
.venv/bin/python tools/fetch_weights.py      # nodo de acceso: solo descarga pesos (los nodos de cómputo van sin conexión)
.venv/bin/python xview_det.py                # split + export a formato YOLO en cache/yolo (~1 min)
.venv/bin/python det_metrics.py              # autotest de la métrica
```

El venv ocupa unos 6 GB, sobre todo PyTorch y sus librerías de CUDA, y cuenta en la cuota de 1 TB del grupo.

## Uso

En el nodo de acceso no se entrena: todo va por SLURM.

```bash
sbatch slurm/gpu_check.sbatch                                          # comprobar la GPU (debug-gpu, 10 min)
sbatch --job-name=EXP-001 --gres=gpu:1 slurm/run.sbatch sweep.py exp001  # un experimento
squeue -u $USER                                                        # seguimiento
sacct -j <jobid> --format=JobID,JobName,State,Elapsed                  # estado al terminar

python tools/compare.py EXP-001_yolo11n EXP-002_yolo11s                # Δ mAP@50 con IC 95 % (CPU)
python tools/infer_sweep.py EXP-001_yolo11n                            # métricas frente al umbral de confianza (CPU)
sbatch --job-name=sub --gres=gpu:1 --time=00:30:00 slurm/run.sbatch predict.py EXP-001_yolo11n --conf 0.05
```

Dónde queda cada salida:
- Logs de SLURM: `experiments/logs/<job>-<id>.out`.
- Métricas: `experiments/experiments.csv` (una fila por run) y `experiments/history/<run_id>.json`.
- Curvas y gráficas de Ultralytics: `experiments/runs/<run_id>/`.
- Para Codabench: `submissions/<EXP>/prediction.zip`, que contiene `prediction.json` en el formato del notebook del profesor. Lo sube un miembro del grupo con su cuenta `@alumnos.upm.es`.
