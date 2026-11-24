"""Experiment definitions, one SLURM job per name (DEC-003): `python sweep.py <name> [<name> ...]`.

Each name maps to (run_id, exp_id, phase, change, cfg). One factor changes with respect to the stated base.
Runs already in experiments/experiments.csv are skipped, so a job can be resubmitted after a failure.
"""
import os
import sys

os.environ.setdefault('YOLO_OFFLINE', '1')
from train_det import DEFAULT_CFG, merge_cfg, run  # noqa: E402

BASE = merge_cfg(DEFAULT_CFG)          # EXP-001: yolo11n COCO, 640, 50 epochs, flips, max_det 1500, conf 0.001

RUNS = {
    # end-to-end check of the pipeline on 5 % of the data (debug-gpu, 20 min)
    'smoke': ('SMOKE_yolo11n', 'SMOKE', 'P0', 'smoke test (5 % train, 2 épocas)',
              merge_cfg(BASE, epochs=2, fraction=0.05, yolo_args={'cache': False, 'workers': 4})),
    'smoke_tv': ('SMOKE_retinanet', 'SMOKE', 'P0', 'smoke test torchvision (5 % train, 1 época)',
                 merge_cfg(BASE, family='tv', model='retinanet_r50_fpn_v2', epochs=1, fraction=0.05, batch=4,
                           tv_args={'workers': 4, 'eval_every': 1})),
    # P1: YOLO, model size (base: EXP-001)
    'exp001': ('EXP-001_yolo11n', 'EXP-001', 'P1', 'YOLO11n COCO, 640, 50 ép. (baseline)', BASE),
    'exp002': ('EXP-002_yolo11s', 'EXP-002', 'P1', 'talla s', merge_cfg(BASE, model='yolo11s')),
    'exp003': ('EXP-003_yolo11m', 'EXP-003', 'P1', 'talla m', merge_cfg(BASE, model='yolo11m')),
    # P2: pretrained COCO vs from scratch (base: EXP-001, the cheapest size)
    'exp004': ('EXP-004_yolo11n_scratch', 'EXP-004', 'P2', 'desde cero (sin COCO)', merge_cfg(BASE, pretrained=False)),
    # P4: classic single-stage detectors from torchvision (COCO), same split and metric
    'exp006': ('EXP-006_ssd300', 'EXP-006', 'P4', 'SSD300-VGG16 COCO, entrada 300 (original)',
               merge_cfg(BASE, family='tv', model='ssd300_vgg16', epochs=24, batch=32,
                         tv_args={'lr': 0.01})),
    'exp007': ('EXP-007_retinanet', 'EXP-007', 'P4', 'RetinaNet-R50-FPN v2 COCO, entrada 800 (original)',
               merge_cfg(BASE, family='tv', model='retinanet_r50_fpn_v2', epochs=24, batch=8,
                         tv_args={'lr': 0.01})),
}


def size_run(size_model):
    """P3: input resolution 1024 on the chosen size (EXP-005)."""
    return ('EXP-005_%s_1024' % size_model, 'EXP-005', 'P3', 'imgsz 1024 (objetos pequeños)',
            merge_cfg(BASE, model=size_model, imgsz=1024, batch=8))


if __name__ == '__main__':
    for name in sys.argv[1:]:
        if name.startswith('exp005_'):
            spec = size_run(name.split('_', 1)[1])
        else:
            spec = RUNS[name]
        run(*spec)
