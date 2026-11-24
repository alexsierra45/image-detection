# LOGBOOK: xview_detection

Detección de 4 clases (Small car, Bus, Truck, Building) en tiles 640×640 de imágenes de satélite de xView [R1].

Bitácora viva del proyecto: todo cambio, experimento, idea, incidencia y decisión, con su justificación teórica. Es la memoria del proyecto: si no está aquí, no pasó. Proceso heredado de `../image-recognition` (DEC-001).

**Convenciones**
- IDs: `EXP-###` experimento · `CHG-###` cambio de código sin efecto en el modelo · `ISS-###` incidencia · `IDEA-###` idea pendiente · `DEC-###` decisión.
- Citas: `[C#]` documentos del curso en [references/](references/README.md) (con página o sección), `[R#]` referencias externas (al final).
- Métricas crudas por run: [experiments/experiments.csv](experiments/experiments.csv) y `experiments/history/<run_id>.json`. Curvas: `experiments/runs/<run_id>/results.csv` (YOLO) o `history.csv` (torchvision). Detecciones de val: `experiments/preds/<run_id>_val.npz` (no versionado).
- **Métricas del challenge** (las del leaderboard y de la celda 19 del notebook del profesor): **mAP@50** (decide), Mean Accuracy, Mean Precision, Mean Recall, siempre sobre el split de validación fijo (DEC-002).
- Un experimento cambia **un solo factor** respecto a su experimento base, para poder atribuirle el efecto.
- Lo que no se ha ejecutado figura como *pendiente*; nunca se anotan resultados de memoria. Estados de un run: código escrito → enviado (id) → terminado → validado.

---

## Estado actual

| | |
|---|---|
| Etapa del curso | **Entrega 1: detectores de una etapa** (cierra el 2026-11-29 a las 23:00, DEC-004) |
| Entorno | Magerit: `.venv` con Python 3.13.5, torch 2.6.0+cu124, Ultralytics 8.4.218 (CHG-001). GPU *pendiente de verificar* (`slurm/gpu_check.sbatch`) |
| Datos | Split por escena fijo (DEC-002): train 6838 / val 768 tiles (55 escenas). Export a YOLO en `cache/yolo` (CHG-002): 481 105 cajas, 7 descartadas, 44 344 recortadas al borde |
| Métrica | Port 1:1 de la del profesor (CHG-003), validado: GT → mAP 1; versión rápida == referencia |
| Mejor experimento | — (ninguno ejecutado) |
| Referencia | Notebook del profesor (KerasCV YOLOv8-xs, 40 épocas, split aleatorio por tile): mAP 19,73 %, mRecall 21,07 %, mPrecision 37,37 %, mAccuracy 33,92 %. No comparable con nuestro split |
| Próximo paso | En cola en standard-gpu desde el 2026-11-24 13:50: EXP-001 (1394533), EXP-002 (1394534), EXP-006 (1394535), EXP-007 (1394536). Smoke tests de YOLO y torchvision validados. Mientras tanto: notebook entregable y pipeline del informe |

## Índice de experimentos

| ID | Fecha | Fase | Cambio respecto a la base | Base | mAP@50 | mRecall | mPrecision | mAcc | Run | Estado |
|----|-------|------|---------------------------|------|--------|---------|------------|------|-----|--------|
| EXP-001 | 2026-11-24 | P1 | YOLO11n COCO, 640, 50 épocas (baseline) | — | | | | | `EXP-001_yolo11n` | enviado (1394533) |
| EXP-002 | — | P1 | Talla s | EXP-001 | | | | | `EXP-002_yolo11s` | enviado (1394534) |
| EXP-003 | — | P1 | Talla m | EXP-001 | | | | | `EXP-003_yolo11m` | pendiente |
| EXP-004 | — | P2 | Desde cero (sin COCO) | EXP-001 | | | | | `EXP-004_yolo11n_scratch` | pendiente |
| EXP-005 | — | P3 | imgsz 1024 | mejor talla | | | | | `EXP-005_*_1024` | pendiente |
| EXP-006 | — | P4 | SSD300-VGG16 COCO (torchvision) | — | | | | | `EXP-006_ssd300` | enviado (1394535) |
| EXP-007 | — | P4 | RetinaNet-R50-FPN v2 COCO (torchvision) | — | | | | | `EXP-007_retinanet` | enviado (1394536) |
| EXP-008 | — | P5 | Inferencia: conf de exportación, NMS, max_det, TTA | mejor | | | | | — | pendiente |

---

## Experimentos

*(ninguno ejecutado todavía)*

---

## Registro de cambios (CHG)

### CHG-001: entorno del proyecto en Magerit e infraestructura copiada de image-recognition (2026-11-24)
- **venv:** módulo `Python/3.13.5-GCCcore-14.3.0` (lleva RPATH: el venv arranca sin el módulo y sirve de kernel de VS Code) + `requirements.txt` fijado:
  - torch 2.6.0+cu124 y torchvision 0.21.0+cu124 (ISS-002);
  - Ultralytics 8.4.218, rasterio 1.4.4, scikit-learn 1.7.2, matplotlib 3.10.9, pandas 2.3.3.
  - Instalado con `TMPDIR` en SCRATCH y `pip cache purge` (lecciones de ISS-015 de recognition).
- **SLURM:**
  - `slurm/run.sbatch`: lanzador genérico; standard-gpu, 1 L40S, 12 CPU × 4 GB, 3 h.
  - `slurm/gpu_check.sbatch` + `tools/gpu_check.py`: torch ve la GPU, conv y NMS en GPU, `cv2.imread` == rasterio en 20 tif.
  - `slurm/env.sh`: `YOLO_CONFIG_DIR`, `TORCH_HOME` en `cache/`, `YOLO_OFFLINE=1`.
- **Pesos offline:** `tools/fetch_weights.py`, que solo descarga y se ejecuta en el nodo de acceso. Motivo: no consta que los nodos de cómputo tengan salida a internet.
- **Copiado de recognition:** `.claude/` (hooks, skills `log-experiment` y `next-ideas`, comando `/new-delivery`), `.githooks/pre-commit`, `tools/{setup.sh,run_notebook.py,nb_textconv.py}`, `plot_theme.py`, `.gitattributes`.
- **Enlaces simbólicos:** `xview_detection` → PROJECT (solo lectura) y `cache` → `SCRATCH/y404034/image-detection-cache` (permisos 700).

### CHG-002: módulo de datos `xview_det.py` (2026-11-24)
- **Anotaciones:** carga en O(N), agrupadas por imagen. El notebook del profesor recorre todas las anotaciones para cada imagen, en O(N·M).
- **Split:** fijo, agrupado por escena (DEC-002).
- **Export a Ultralytics** en `cache/yolo/`:
  - Las imágenes son symlinks a los tif de PROJECT, sin copia y sin gastar cuota.
  - Labels `cls xc yc w h` normalizados.
  - **Cajas recortadas al tile** y descartadas si quedan con < 1 px de ancho o alto (ISS-003). Este recorte solo afecta a los targets de entrenamiento: la evaluación usa siempre el GT crudo, como Codabench.
- **Formato del challenge:** `to_submission`, idéntico a la celda 23 del profesor. Escribe las detecciones ordenadas por clase y por confianza descendente (ISS-001).
- **Detecciones crudas en npz:** `save_dets` / `load_dets`, para reevaluar sin GPU.
- **Ejecutado (2026-11-24, `python xview_det.py`, 11 s de CPU):**
  - Split: train 6838 / val 768 tiles; 55 de 581 escenas en val. Desviación máxima de las proporciones respecto al 10 %: 0,46 pp.
  - Cajas en val: Small car 18 611, Bus 598, Truck 1085, Building 26 655.
  - Export (`cache/yolo/export_stats.json`): de 481 112 cajas se escriben 481 105; 7 se descartan por degeneradas y 44 344 se recortan al borde. Test: 852 symlinks.

### CHG-003: métrica del challenge, `det_metrics.py` (2026-11-24)
- `reference_evaluate`: port literal de las celdas 16-19 del profesor.
- `match` + `metrics`: la misma lógica vectorizada por imagen, con un registro por detección. Permite filtrar por `conf` y `max_det` sin volver a casar, y hacer bootstrap pareado por imagen (`bootstrap_delta`).
- Reglas de la referencia que se conservan:
  - IoU con la convención +1 px de VOC [R6].
  - TP solo si el GT de máxima IoU sigue libre.
  - AP con interpolación en todos los puntos.
  - Las tres métricas macro incluyen BACKGROUND en el promedio.
- Autotest (`python det_metrics.py`):
  - GT como predicción → mAP = 1;
  - predicciones ruidosas → la versión rápida coincide con la referencia hasta 1e-9.
- **Validado (2026-11-24)**, en el nodo de acceso (9 s de CPU):
  - GT como predicción: las 4 métricas valen 1,0000.
  - Con 60 imágenes de val y predicciones ruidosas, las dos versiones dan lo mismo en las 4 métricas: mAP50 0,158615, mAcc 0,295741, mPrec 0,222019, mRec 0,350021.

### CHG-004: `train_det.py`, `sweep.py` y `predict.py` (2026-11-24)
Flujo de un run:
1. Entrenar.
2. Predecir val con la configuración `infer` del run.
3. Calcular las métricas del challenge.
4. Escribir `history/<run_id>.json`, `preds/<run_id>_val.npz` y una fila de `experiments.csv`. La escritura del CSV va con lock, porque varios trabajos pueden terminar a la vez (ISS-016 de recognition).

Otros detalles:
- `sweep.py <nombre>`: un trabajo de SLURM por experimento. Se salta los runs ya registrados.
- `predict.py <run_id>` genera `submissions/<EXP>/prediction.{json,zip}` y comprueba el esquema.
- **Smoke YOLO validado (2026-11-24, trabajo 1394523, debug-gpu, A100, r2n3, 1:47):** `SMOKE_yolo11n` (5 % de train = 342 tiles, 2 épocas, 4 workers, sin caché).
  - 18 s por época, validación de 768 tiles incluida. Postproceso de 19,8 ms por imagen, sin avisos de `NMS time limit`.
  - Ultralytics encuentra las labels de los symlinks (`labels/train.cache` creado). AMP check superado sin conexión.
  - Val con la métrica del curso: mAP@50 0,085 (Ultralytics: 0,096). La diferencia viene del GT sin recortar, la IoU +1 px y la interpolación.
  - `n_det` = 768 × 1500: con `conf` 0,001, **todas las imágenes llegan al tope de `max_det`** (IDEA-001 / EXP-008).
  - Se quita `half=True` de `predict_yolo`: en Ultralytics 8.4 está deprecado y emitía un aviso por lote.
- **Smoke torchvision validado (trabajo 1394524, A100, 42 s):** `SMOKE_retinanet` (5 %, 1 época, batch 4) entrena a unas 38 img/s. Pérdidas de train y val, mAP@50 de val y `best.pt` funcionan; mAP@50 de val 0,039.
- El driver de r2n3 es ahora el **615.71.10** (el 2026-10-23 era el 545).

### CHG-005: detectores de torchvision, `tv_detectors.py` (2026-11-24)
- Modelos: SSD300-VGG16, SSDlite320 y RetinaNet-R50-FPN v2 con pesos COCO. Se sustituye la cabeza de clasificación por 4 + 1 clases (torchvision reserva la etiqueta 0 para el fondo).
- Entrenamiento: SGD con momentum, warmup lineal y coseno, AMP y clip de gradiente a 10. Aumentos: volteos horizontal y vertical.
- En cada época se calcula la pérdida de val (en modo train, sin gradiente y con el momentum de BN a 0). Cada `eval_every` épocas se calcula el mAP@50 de val, que elige `best.pt`.
- La inferencia usa `score_thresh = conf`, `detections_per_img = max_det` y `topk_candidates` alto: las imágenes tienen hasta 1029 objetos.
- **Bug corregido antes de encolar:** `retrieve_out_channels` está en `torchvision.models.detection._utils`, no en `ssd._utils`. Se encontró con una prueba en CPU en el nodo de acceso (construcción + un paso de train y eval con 1 imagen): SSD300 da pérdidas finitas y 1500 cajas en eval.

### CHG-007: herramientas de análisis sin GPU (2026-11-24)
- `tools/compare.py <base> <new>`: las 4 métricas de los dos runs y un bootstrap pareado por imagen de val del Δ mAP@50 (500 réplicas, IC 95 %). Guarda el resultado en `experiments/compare/`.
- `tools/infer_sweep.py <run>`: las 4 métricas en función del `conf` de exportación (0,001…0,5) y de `max_det` (300/1000/1500), a partir de `preds/<run>_val.npz`, sin volver a casar las detecciones (EXP-008, IDEA-001).
- `slurm/env.sh` añade `OPENCV_LOG_LEVEL=ERROR`: OpenCV emite 5 avisos por cada tif por las etiquetas GeoTIFF que no reconoce.

---

## Incidencias (ISS)

### ISS-001: el evaluador del profesor desalinea confianzas y cajas si las predicciones no vienen ordenadas · mitigada (→ CHG-002)
En la celda 18, `pred_boxes` se reordena por confianza descendente, pero la confianza que se guarda es `pred_confidences[pred_idx]`, que es el índice de la lista **sin** ordenar. Si las detecciones de una imagen y clase no llegan ya ordenadas, cada TP/FP queda asociado a la confianza de otra caja, la curva PR se desordena y el AP baja.

Probablemente Codabench usa el mismo código. Por eso `to_submission` escribe siempre las detecciones ordenadas por confianza descendente dentro de cada clase; así el `argsort` es la identidad y no hay desalineación. `det_metrics.reference_evaluate` conserva el bug a propósito.

### ISS-002: no hay torchvision cu121 para Python 3.13 · resuelta (→ CHG-001)
- El primer intento de instalar `torch==2.5.1+cu121` + `torchvision==0.20.1+cu121` falló: el índice cu121 no tiene torchvision 0.20.1 para cp313.
- Se pasa a torch 2.6.0+cu124 y torchvision 0.21.0+cu124.
- El driver de los nodos (545, CUDA 12.3) ejecuta el runtime 12.4 por compatibilidad entre versiones menores de CUDA 12, igual que TF 2.21 con CUDA 12.5/12.9 en recognition (CHG-029).
- **Confirmado (2026-11-24, trabajo 1394522, A100-PCIE-40GB, r2n3):** `cuda.is_available`, runtime 12.4, cuDNN 9.1, conv 3×3 (20× 16×3×640×640) en 0,40 s, autocast fp16 y NMS de torchvision en GPU, todo correcto.
  - El trabajo terminó en FAILED por un bug de rutas en la comprobación de imágenes de `tools/gpu_check.py` (anteponía `xview_detection/` dos veces). Ya está corregido.
  - Esa comprobación (solo lectura de 20 tif, sin GPU) se repitió en el nodo de acceso: `cv2.imread` == rasterio (con RGB↔BGR), 640×640×3 uint8.

### ISS-003: cajas de GT fuera del tile y degeneradas · abierta (mitigada en el entrenamiento)
- Unas 25 000 cajas se salen del tile 640×640 (x1 hasta −741, x2 hasta 1676) y 5 tienen ancho ≤ 0.
- Para entrenar se recortan (CHG-002).
- Para evaluar se usa el GT crudo, como hará Codabench. Consecuencia: una detección perfecta de la parte visible de un objeto cortado por el borde puede tener IoU < 0,5 con su GT y contar como FP + FN. Es un techo del mAP que afecta a todos los modelos por igual.

### ISS-004: standard-gpu saturada a 5 días de la entrega · abierta
- **Síntoma (2026-11-24, 13:45):** `sbatch --test-only` estima el arranque en standard-gpu para el **2026-12-05/06**, sea cual sea la forma del trabajo (1 GPU de cualquier tipo, 4 u 8 CPU, de 30 min a 3 h). La entrega es el 29/11.
- **Contexto:**
  - standard-gpu y debug-gpu comparten los mismos 5 nodos (`r2n[3-4,6,10-11]`, 392 CPU).
  - debug-gpu arranca enseguida, pero la QoS `debug` limita los minutos de GPU por trabajo: con `--time=01:00:00` salta `QOSMaxGRESMinutesPerJob`, así que el máximo real son unos 20 min.
  - No hay reservas (`scontrol show reservation`).
  - No vemos los trabajos de otros usuarios, así que no se puede saber la longitud real de la cola.
- **Lectura:** la estimación de `--test-only` es el peor caso, porque supone que todos los trabajos agotan su `--time`. El arranque real puede llegar antes.
- **Mitigación (decisión del usuario):** encolar ya lo esencial con presupuestos reducidos (DEC-005). Encadenar trabajos de debug para entrenar queda descartado: sería eludir el límite de la QoS.
- **Feedback para el informe:** es el principal problema de usar Cesvima en esta entrega.

---

## Ideas pendientes (backlog)

### IDEA-001: umbral de confianza de exportación · prioridad alta
- **Qué:** barrer `conf` (0,001…0,5) sobre las detecciones de val guardadas, sin reentrenar, y elegir el del envío (EXP-008).
- **Por qué:** el mAP@50 premia exportar muchas detecciones de baja confianza, porque solo añaden cola a la curva PR [R6]. Mean Precision y Mean Accuracy, en cambio, cuentan todas las exportadas como FP o TP, así que bajan.
- **Coste:** segundos (filtrado de `Records`).
- **Éxito:** un `conf` que mantenga el mAP@50 a < 0,2 pp del máximo y mejore las otras tres.

### IDEA-002: mayor resolución de entrada · prioridad alta (→ EXP-005)
- **Qué:** reescalar los tiles a 1024.
- **Por qué:** la mediana de los objetos es de unos 24×22 px. Con stride 8 en P3, un coche ocupa unas 3×3 celdas. Reescalar mejora la detección de objetos pequeños [R3, R7].
- **Coste:** unas 2,5 veces más por época.

### IDEA-003: desbalance Bus/Truck · prioridad media
- **Qué:** sobremuestrear los tiles que contienen Bus o Truck, o ponderar la pérdida de clase.
- **Por qué:** Building tiene 276k cajas y Bus 6,3k; el AP de Truck en el ejemplo del profesor es 0,36 %. La focal loss [R3] ya mitiga el desbalance fondo/objeto, pero no el desbalance entre clases.

---

## Decisiones (DEC)

### DEC-001: Ultralytics (YOLO) + torchvision (SSD, RetinaNet) en PyTorch, con el proceso de image-recognition (2026-11-24)
Decisión del usuario.
- **Por qué:** el enunciado recomienda KerasCV o Ultralytics. Pero KerasCV está congelado y su sucesor, keras-hub, apenas trae detectores de una etapa. torchvision aporta SSD [R2] y RetinaNet [R3] con pesos COCO, y para la entrega 2 aporta también Faster R-CNN. Todo en un solo venv, con la misma métrica y el mismo split.
- **Alternativa descartada:** seguir en TF/Keras como en recognition (más código propio y peor soporte).
- **Infraestructura replicada:** LOGBOOK, hooks, skills, SLURM, `experiments.csv`, `submissions/` y el pipeline `make bundle`.

### DEC-002: validación agrupada por escena, 10 % de las escenas, fija (2026-11-24)
- **Por qué no un split aleatorio por tile** (el del profesor, `train_test_split(test_size=0.1)`): los tiles de una misma escena son vecinos y comparten iluminación, fecha y estilo urbano, y es probable que algunos se solapen. Un split aleatorio por tile mete en val información de las escenas de train y sobreestima el mAP.
- Train y test del challenge no comparten escenas (581 frente a 68), y la validación debe imitar esa condición.
- **Cómo se eligen las escenas de val:** entre 200 barajados con semilla 0, el que deja en val una proporción de imágenes y de cajas de cada clase más próxima al 10 %. Solo se miran las etiquetas, nunca un modelo.
- El split queda versionado en `experiments/splits/val_scenes.json` y todos los EXP lo usan.

### DEC-003: protocolo experimental de la entrega 1 (2026-11-24)
- **Selección:** un hold-out (DEC-002), una semilla y el mAP@50 de val con la métrica del challenge.
  - No hay validación cruzada: con 5 días y entrenamientos de horas, no cabe.
  - Las diferencias de ≲ 1 pp no son concluyentes sin el IC del bootstrap pareado (`tools/compare.py`).
- **Barrido:** un trabajo de SLURM por EXP con `sweep.py`, hasta 8 GPU en paralelo (QoS `standard`).
- **Configuración común de inferencia**, igual para todos los modelos: `conf` 0,001, `max_det` 1500 (máximo de objetos por imagen = 1029, p99 = 334) y NMS IoU 0,7 (por defecto en Ultralytics).
- **Test del challenge:** solo para el envío final, nunca para decidir.

### DEC-004: calendario (2026-11-24)
La entrega 1 cierra el 2026-11-29 a las 23:00. La entrega es la presentación (PDF, unos 10 minutos), que sale del mismo `make bundle` que el informe; además se sube al menos un modelo a Codabench.

| Fecha | Trabajo |
|---|---|
| 24–25/11 | Entorno, datos, métrica y smoke test |
| 25–26/11 | EXP-001..007 |
| 27/11 | Inferencia, envío y notebook |
| 28/11 | Informe y slides |
| 29/11 | Margen |

### DEC-005: presupuesto reducido y experimentos esenciales primero (2026-11-24)
Por ISS-004 (cola de standard-gpu estimada hasta el 5/12), el usuario elige encolar ya solo lo esencial para ganar prioridad:

| Experimento | Épocas | Comentario |
|---|---|---|
| EXP-001 (YOLO11n) | 50 (antes 60) | |
| EXP-002 (YOLO11s) | 50 | |
| EXP-006 (SSD300) | 24 | |
| EXP-007 (RetinaNet) | 24 | |

- EXP-003, EXP-004 y EXP-005 se envían solo si los primeros arrancan a tiempo.
- `--gres=gpu:1` sin tipo, para no restringir la cola. Riesgo: una V100 de 16 GB sería más lenta, de ahí el margen de `--time`. La métrica no depende de la GPU.
- Velocidades de los smoke: YOLO11n ~18 s por época con el 5 % sin caché; RetinaNet ~38 img/s.

---

## Referencias externas

- [R1] Lam, D. et al. (2018). *xView: Objects in Context in Overhead Imagery*. arXiv:1802.07856.
- [R2] Liu, W. et al. (2016). *SSD: Single Shot MultiBox Detector*. ECCV.
- [R3] Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P. (2017). *Focal Loss for Dense Object Detection* (RetinaNet). ICCV.
- [R4] Redmon, J., Divvala, S., Girshick, R., Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection*. CVPR.
- [R5] Jocher, G., Qiu, J. (2024). *Ultralytics YOLO11* (software). https://github.com/ultralytics/ultralytics
- [R6] Everingham, M. et al. (2010). *The PASCAL Visual Object Classes (VOC) Challenge*. IJCV 88.
- [R7] Lin, T.-Y. et al. (2017). *Feature Pyramid Networks for Object Detection*. CVPR.
- [R8] Lin, T.-Y. et al. (2014). *Microsoft COCO: Common Objects in Context*. ECCV.

---

## Plantillas

<details>
<summary>EXP</summary>

```markdown
### EXP-###: <título>
- **Fecha:** AAAA-MM-DD · **Fase:** P# · **Base:** EXP-### · **Estado:** pendiente / enviado (job) / ejecutado / descartado
- **Hipótesis:** qué se espera y por qué.
- **Cambio:** el único factor que cambia respecto a la base.
- **Fundamento:** teoría con cita [C#, p. X] / [R#].
- **Configuración:** solo lo que difiere de la base; recursos SLURM.
- **Resultados:** run `<run_id>` (job) · mAP@50 · mRecall · mPrecision · mAcc · AP por clase · época del mejor modelo · s/época · observaciones de las curvas (train/val loss) y de la matriz de confusión.
- **Comparación:** Δ mAP@50 frente a la base con IC 95 % (`tools/compare.py`).
- **Conclusión:** ¿se confirma la hipótesis? ¿por qué sí o no?
- **Siguiente paso:**
```
</details>

<details>
<summary>CHG / ISS / IDEA / DEC</summary>

```markdown
### CHG-###: <título> (AAAA-MM-DD)
Qué cambió, dónde y por qué. Si afecta al modelo, es un EXP.

### ISS-###: <título> · abierta / resuelta (→ CHG/EXP-###)
Síntoma, causa y evidencia. Resolución.

### IDEA-###: <título> · prioridad alta/media/baja
- **Qué:** · **Por qué:** con cita · **Coste/riesgo:** · **Éxito:**

### DEC-###: <título> (AAAA-MM-DD)
Qué se decidió, alternativas descartadas y por qué.
```
</details>
