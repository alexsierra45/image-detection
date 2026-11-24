---
name: next-ideas
description: Analiza el estado del proyecto xview_detection (LOGBOOK.md, runs, curvas, matriz de confusión, notebook, references/) y propone los siguientes experimentos priorizados, cada uno con hipótesis, fundamento teórico citado, coste estimado en GPU-horas y criterio de éxito. Usar cuando el usuario pregunte qué probar a continuación o empiece un tema nuevo del curso.
argument-hint: "[tema del curso actual, opcional]"
---

# Proponer los siguientes experimentos

Tema o foco indicado por el usuario (puede estar vacío): $ARGUMENTS

## 1. Diagnóstico con evidencia
- LOGBOOK.md: estado actual, índice, ideas e incidencias abiertas, y conclusiones de los últimos EXP.
- Mejor run de `experiments/experiments.csv` (mAP@50 de val) y su `history/<id>.json`:
  - Curvas: ¿underfitting, overfitting u optimización inestable? ¿Se cortó el entrenamiento antes de converger?
  - AP por clase y matriz de confusión (con BACKGROUND): ¿el problema son FN (recall bajo: objetos pequeños, Truck/Bus raros), FP (precisión baja) o confusiones entre vehículos?
- Estado de la cola de GPU (ISS-004): proponer experimentos que quepan en el presupuesto real.
- `references/`: si hay documentos nuevos del curso (el tema que se está viendo), priorizar ideas que lo pongan en práctica. Si no consta en qué tema va el usuario, preguntarlo.

## 2. Proponer de 3 a 5 experimentos
Ordenados por impacto esperado / coste. Para cada uno:
- **Hipótesis** y qué síntoma del diagnóstico ataca.
- **Cambio concreto**: un solo factor respecto al mejor EXP.
- **Fundamento** con cita (`[C#, p. X]` de `references/` o `[R#]`). Si es heurística, decirlo.
- **Coste estimado** en GPU (s/época del run base, épocas, cola) y riesgo.
- **Criterio de éxito** medible (p. ej. +X pp de mAP@50 con IC 95 % que excluya 0).

Antes de proponer, revisar si alguna IDEA del backlog ya cubre la propuesta y reutilizarla.

## 3. Cerrar
Preguntar cuáles añadir al backlog como IDEA-### (o repriorizar las existentes) y registrarlas en LOGBOOK.md. No modificar el notebook sin confirmación del usuario.
