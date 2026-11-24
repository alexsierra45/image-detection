"""One experiment run = train a detector, predict the val split, score it with the challenge metric, log it (CHG-004).

Outputs (one run_id):
- experiments/runs/<run_id>/            Ultralytics run dir (results.csv, plots, weights/best.pt) or the torchvision
                                        equivalent (history.csv, best.pt)
- experiments/preds/<run_id>_val.npz    raw val detections (re-threshold / bootstrap without GPU; gitignored)
- experiments/history/<run_id>.json     config, per-epoch history, metrics, confusion matrix, SLURM info
- experiments/experiments.csv           one row per run (file lock: several jobs may finish at once)
"""
import copy
import csv
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
from filelock import FileLock

import det_metrics
from xview_det import CATEGORIES, DATA, ROOT, YOLO_DIR, get_split, load_annotations

EXP_DIR = ROOT / 'experiments'
RUNS_DIR = EXP_DIR / 'runs'
PREDS_DIR = EXP_DIR / 'preds'
HIST_DIR = EXP_DIR / 'history'
CSV_PATH = EXP_DIR / 'experiments.csv'
WEIGHTS = ROOT / 'cache' / 'weights'

DEFAULT_CFG = {
    'family': 'yolo',            # yolo | tv (torchvision)
    'model': 'yolo11n',          # yolo: yolo11{n,s,m}; tv: ssd300_vgg16 | ssdlite320 | retinanet_r50_fpn_v2
    'pretrained': True,          # COCO weights
    'imgsz': 640,
    'epochs': 50,                # DEC-005: presupuesto reducido por la cola (ISS-004)
    'batch': 16,
    'seed': 0,
    'fraction': 1.0,             # share of the train split (smoke test)
    'data': 'xview_det.yaml',    # or xview_det_trainval.yaml (final refit, no honest val)
    # Ultralytics training hyper-parameters (defaults of 8.4 unless stated)
    'yolo_args': {'optimizer': 'auto', 'cos_lr': False, 'close_mosaic': 10, 'patience': 100,
                  'flipud': 0.5,     # aerial view has no canonical "up"; professor's notebook flips both axes too
                  'workers': 10, 'cache': 'ram'},
    # torchvision training hyper-parameters
    'tv_args': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4, 'warmup_iters': 1000, 'workers': 10,
                'min_size': None, 'anchor_scales': None, 'eval_every': 2},
    # inference used for val metrics and for the submission
    'infer': {'conf': 0.001, 'iou': 0.7, 'max_det': 1500, 'augment': False},
}


def merge_cfg(base=None, **changes):
    cfg = copy.deepcopy(base or DEFAULT_CFG)
    for k, v in changes.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def done_ids():
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH) as ifs:
        return {r['id'] for r in csv.DictReader(ifs)}


# -----------------------------------------------------------------------------------------------------------------
# Ultralytics YOLO
# -----------------------------------------------------------------------------------------------------------------
def train_yolo(run_id, cfg):
    from ultralytics import YOLO
    if cfg['pretrained']:
        model = YOLO(str(WEIGHTS / (cfg['model'] + '.pt')))
    else:
        model = YOLO(cfg['model'] + '.yaml')
    a = cfg['yolo_args']
    model.train(data=str(YOLO_DIR / cfg['data']), epochs=cfg['epochs'], imgsz=cfg['imgsz'], batch=cfg['batch'],
                seed=cfg['seed'], deterministic=True, fraction=cfg['fraction'], pretrained=cfg['pretrained'],
                project=str(RUNS_DIR), name=run_id, exist_ok=True, max_det=cfg['infer']['max_det'],
                amp=True, plots=True, verbose=True, **a)
    run_dir = RUNS_DIR / run_id
    best = run_dir / 'weights' / 'best.pt'
    history = []
    if (run_dir / 'results.csv').exists():
        with open(run_dir / 'results.csv') as ifs:
            history = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(ifs)]
    return best, history


def predict_yolo(weights, files, infer, imgsz, batch=16):
    """{filename: (boxes xyxy, scores, labels)} for files relative to DATA."""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    dets = {}
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        results = model.predict([str(DATA / f) for f in chunk], imgsz=imgsz, conf=infer['conf'], iou=infer['iou'],
                                max_det=infer['max_det'], augment=infer.get('augment', False),
                                verbose=False, batch=len(chunk))
        for f, r in zip(chunk, results):
            b = r.boxes
            dets[f] = (b.xyxy.cpu().numpy().astype(np.float64), b.conf.cpu().numpy().astype(np.float64),
                       b.cls.cpu().numpy().astype(np.int64))
    return dets


# -----------------------------------------------------------------------------------------------------------------
def slurm_info():
    keys = ['SLURM_JOB_ID', 'SLURM_JOB_NAME', 'SLURM_JOB_PARTITION', 'SLURM_JOB_GPUS', 'SLURM_CPUS_PER_TASK']
    info = {k: os.environ.get(k) for k in keys}
    info['host'] = socket.gethostname()
    try:
        import torch
        info['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        info['gpu'] = None
    return info


def run(run_id, exp_id, phase, change, cfg, notes='', skip_done=True):
    if skip_done and run_id in done_ids():
        print('skip (ya en experiments.csv):', run_id)
        return None
    for d in (RUNS_DIR, PREDS_DIR, HIST_DIR):
        d.mkdir(parents=True, exist_ok=True)
    print('=== %s (%s) %s\n%s' % (run_id, exp_id, change, json.dumps(cfg, indent=1)), flush=True)
    t0 = time.time()
    if cfg['family'] == 'yolo':
        best, history = train_yolo(run_id, cfg)
        t_train = time.time() - t0
        split = get_split()
        dets = predict_yolo(best, split['val'], cfg['infer'], cfg['imgsz'])
    else:
        import tv_detectors
        best, history = tv_detectors.train(run_id, cfg, RUNS_DIR / run_id)
        t_train = time.time() - t0
        split = get_split()
        dets = tv_detectors.predict(best, cfg, split['val'])
    t_pred = time.time() - t0 - t_train
    from xview_det import save_dets
    save_dets(PREDS_DIR / (run_id + '_val.npz'), dets, split['val'])
    gt = load_annotations()
    m = det_metrics.evaluate(dets, gt, split['val'])
    print('VAL %s: mAP50 %.4f | mAcc %.4f | mPrec %.4f | mRec %.4f | AP %s' % (
        run_id, m['mAP50'], m['mean_accuracy'], m['mean_precision'], m['mean_recall'],
        {c: round(m['AP_' + c], 4) for c in CATEGORIES.values()}), flush=True)

    best_epoch, ul = None, {}
    if history and cfg['family'] == 'yolo':
        key = 'metrics/mAP50(B)'
        vals = [h.get(key, np.nan) for h in history]
        if not all(np.isnan(vals)):
            i = int(np.nanargmax([0.1 * h.get('metrics/mAP50(B)', 0) + 0.9 * h.get('metrics/mAP50-95(B)', 0) for h in history]))
            best_epoch = int(history[i]['epoch'])
            ul = {'ul_mAP50': history[i].get('metrics/mAP50(B)'), 'ul_mAP50_95': history[i].get('metrics/mAP50-95(B)')}
    elif history:
        i = int(np.nanargmax([h.get('val_mAP50', np.nan) for h in history]))
        best_epoch = int(history[i]['epoch'])
    row = {
        'id': run_id, 'exp_id': exp_id, 'phase': phase, 'change': change, 'family': cfg['family'], 'model': cfg['model'],
        'pretrained': cfg['pretrained'], 'imgsz': cfg['imgsz'], 'epochs': cfg['epochs'], 'actual_epochs': len(history),
        'best_epoch': best_epoch,
        'mAP50': round(m['mAP50'], 5), 'mean_accuracy': round(m['mean_accuracy'], 5),
        'mean_precision': round(m['mean_precision'], 5), 'mean_recall': round(m['mean_recall'], 5),
        **{'AP_' + c: round(m['AP_' + c], 5) for c in CATEGORIES.values()},
        'ul_mAP50': ul.get('ul_mAP50'), 'ul_mAP50_95': ul.get('ul_mAP50_95'),
        'n_det': m['n_det'], 'infer_conf': cfg['infer']['conf'], 'infer_max_det': cfg['infer']['max_det'],
        'time_train_s': round(t_train), 's_per_epoch': round(t_train / max(len(history), 1), 1), 'time_pred_s': round(t_pred),
        'gpu': None, 'job': os.environ.get('SLURM_JOB_ID'), 'notes': notes, 'config': json.dumps(cfg, sort_keys=True),
    }
    info = slurm_info()
    row['gpu'] = info['gpu']
    with open(HIST_DIR / (run_id + '.json'), 'w') as ofs:
        json.dump({'row': row, 'config': cfg, 'metrics': m, 'history': history, 'slurm': info,
                   'weights': str(best)}, ofs, indent=1, default=float)
    with FileLock(str(CSV_PATH) + '.lock'):
        new = not CSV_PATH.exists()
        if not new:   # union of columns, so adding a column later never breaks the file
            with open(CSV_PATH) as ifs:
                old_rows = list(csv.DictReader(ifs))
            fields = list(old_rows[0].keys()) if old_rows else []
            extra = [k for k in row if k not in fields]
            if extra:
                fields += extra
                with open(CSV_PATH, 'w', newline='') as ofs:
                    w = csv.DictWriter(ofs, fieldnames=fields)
                    w.writeheader()
                    w.writerows(old_rows)
        else:
            fields = list(row)
        with open(CSV_PATH, 'a', newline='') as ofs:
            w = csv.DictWriter(ofs, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow(row)
    print('RUN DONE %s in %.0f s' % (run_id, time.time() - t0), flush=True)
    return row
