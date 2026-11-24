"""xview_detection data: annotations, scene-grouped train/val split, YOLO export and the challenge JSON format.

Facts about the data (LOGBOOK, CHG-002):
- xview_det_train.json: dicts keyed by string index; bbox is absolute xyxy; category_id is the class *name*.
- Boxes are not clipped to the 640x640 tile (some go from -741 to 1676) and 5 are degenerate.
- Image names are '<scene>_<uuid>.tif': tiles of the same scene overlap in content, so the validation split
  groups by scene (DEC-002) instead of the professor's random per-tile split.
"""
import json
import os
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'xview_detection'                    # -> /home/y404/PROJECT/xview_detection (read-only)
CACHE = ROOT / 'cache'                             # -> SCRATCH/y404034/image-detection-cache
YOLO_DIR = CACHE / 'yolo'
SPLIT_FILE = ROOT / 'experiments' / 'splits' / 'val_scenes.json'

CATEGORIES = {0: 'Small car', 1: 'Bus', 2: 'Truck', 3: 'Building'}   # same order as the professor's notebook
NAME2ID = {v: k for k, v in CATEGORIES.items()}
TILE = 640


def load_geoimage(filename):
    """Professor's loader (cell 4): HxWxC array in RGB, via rasterio. `filename` relative to DATA or absolute."""
    import rasterio
    warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
    path = filename if os.path.isabs(filename) else DATA / filename
    with rasterio.open(path, 'r') as src:
        img = np.zeros((src.height, src.width, src.count), dtype=src.profile['dtype'])
        for band in range(src.count):
            img[:, :, band] = src.read(band + 1)
    return img


def scene_of(image_id):
    return os.path.basename(image_id).split('_')[0]


def load_annotations(json_file=DATA / 'xview_det_train.json'):
    """{filename ('xview_train/<f>.tif'): {'width', 'height', 'boxes': (N,4) raw xyxy float, 'labels': (N,) int}}.

    Raw boxes (not clipped): this is the ground truth the challenge evaluates against."""
    with open(json_file) as ifs:
        data = json.load(ifs)
    by_id = defaultdict(list)
    for ann in data['annotations'].values():
        by_id[ann['image_id']].append(ann)
    out = {}
    for img in data['images'].values():
        anns = by_id.get(img['image_id'], [])
        out[img['filename']] = {
            'width': img['width'], 'height': img['height'],
            'boxes': np.array([a['bbox'] for a in anns], dtype=np.float64).reshape(-1, 4),
            'labels': np.array([NAME2ID[a['category_id']] for a in anns], dtype=np.int64),
        }
    return out


def test_filenames():
    """Test tiles as the professor's notebook names them ('xview_test/<f>.tif'), sorted for determinism."""
    return sorted('xview_test/' + f for f in os.listdir(DATA / 'xview_test') if f.endswith('.tif'))


# --------------------------------------------------------------------------------------------------------------
# Train/val split grouped by scene (DEC-002)
# --------------------------------------------------------------------------------------------------------------
def _class_counts(anns, files):
    c = Counter()
    for f in files:
        c.update(anns[f]['labels'].tolist())
    return np.array([c[k] for k in CATEGORIES], dtype=float)


def make_split(anns, val_frac=0.10, n_candidates=200, seed=0):
    """Pick the val scenes among `n_candidates` random scene shuffles (fixed seed), keeping the one whose share
    of images and of every class's boxes is closest to `val_frac`. Uses only the labels, never a model."""
    scenes = sorted({scene_of(f) for f in anns})
    files_of = defaultdict(list)
    for f in anns:
        files_of[scene_of(f)].append(f)
    total = _class_counts(anns, anns)
    rng = random.Random(seed)
    best = None
    for _ in range(n_candidates):
        order = scenes[:]
        rng.shuffle(order)
        val, n = [], 0
        for s in order:
            if n >= val_frac * len(anns):
                break
            val.append(s)
            n += len(files_of[s])
        vfiles = [f for s in val for f in files_of[s]]
        share = _class_counts(anns, vfiles) / total
        dev = max(abs(share - val_frac).max(), abs(len(vfiles) / len(anns) - val_frac))
        if best is None or dev < best[0]:
            best = (dev, sorted(val))
    val_scenes = set(best[1])
    train = sorted(f for f in anns if scene_of(f) not in val_scenes)
    val = sorted(f for f in anns if scene_of(f) in val_scenes)
    vc, tc = _class_counts(anns, val), _class_counts(anns, train)
    return {'val_frac': val_frac, 'seed': seed, 'n_candidates': n_candidates, 'max_share_deviation': best[0],
            'val_scenes': sorted(val_scenes), 'train': train, 'val': val,
            'counts': {'train': dict(zip(CATEGORIES.values(), tc.astype(int).tolist())),
                       'val': dict(zip(CATEGORIES.values(), vc.astype(int).tolist()))}}


def get_split(anns=None):
    """Fixed split from experiments/splits/val_scenes.json (created once, versioned in git)."""
    if SPLIT_FILE.exists():
        with open(SPLIT_FILE) as ifs:
            return json.load(ifs)
    anns = anns if anns is not None else load_annotations()
    split = make_split(anns)
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_FILE, 'w') as ofs:
        json.dump(split, ofs, indent=1)
    return split


# --------------------------------------------------------------------------------------------------------------
# Training targets: clipped boxes (CHG-002)
# --------------------------------------------------------------------------------------------------------------
def clip_boxes(boxes, labels, w=TILE, h=TILE, min_size=1.0):
    """Clip xyxy boxes to the tile and drop those thinner than `min_size` px after clipping.
    Returns (boxes, labels, n_dropped). Used only for training targets; evaluation keeps the raw GT."""
    b = boxes.copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    keep = ((b[:, 2] - b[:, 0]) >= min_size) & ((b[:, 3] - b[:, 1]) >= min_size)
    return b[keep], labels[keep], int((~keep).sum())


def export_yolo(anns=None, split=None, out=YOLO_DIR):
    """Ultralytics layout under cache/yolo: images/{train,val,test}/ are symlinks to the read-only tifs,
    labels/{train,val}/ hold 'cls xc yc w h' normalised. Writes xview_det.yaml (train), xview_det_trainval.yaml
    (train+val, for a final refit) and export_stats.json. Idempotent."""
    anns = anns if anns is not None else load_annotations()
    split = split if split is not None else get_split(anns)
    stats = {'boxes_raw': 0, 'boxes_written': 0, 'dropped': 0, 'clipped': 0}
    parts = {'train': split['train'], 'val': split['val'], 'test': test_filenames()}
    for part, files in parts.items():
        (out / 'images' / part).mkdir(parents=True, exist_ok=True)
        if part != 'test':
            (out / 'labels' / part).mkdir(parents=True, exist_ok=True)
        for f in files:
            name = os.path.basename(f)
            link = out / 'images' / part / name
            if not link.is_symlink():
                link.symlink_to(DATA / f)
            if part == 'test':
                continue
            a = anns[f]
            b, l, dropped = clip_boxes(a['boxes'], a['labels'], a['width'], a['height'])
            stats['boxes_raw'] += len(a['boxes'])
            stats['boxes_written'] += len(b)
            stats['dropped'] += dropped
            outside = ((a['boxes'] < 0).any(1) | (a['boxes'][:, [0, 2]] > a['width']).any(1)
                       | (a['boxes'][:, [1, 3]] > a['height']).any(1))
            stats['clipped'] += int(outside.sum())
            w, h = a['width'], a['height']
            lines = ['%d %.6f %.6f %.6f %.6f' % (c, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)
                     for (x1, y1, x2, y2), c in zip(b, l)]
            with open(out / 'labels' / part / (os.path.splitext(name)[0] + '.txt'), 'w') as ofs:
                ofs.write('\n'.join(lines) + ('\n' if lines else ''))
    names = '\n'.join('  %d: %s' % (k, v) for k, v in CATEGORIES.items())
    for yaml_name, train in [('xview_det.yaml', 'images/train'), ('xview_det_trainval.yaml', '[images/train, images/val]')]:
        with open(out / yaml_name, 'w') as ofs:
            ofs.write('path: %s\ntrain: %s\nval: images/val\ntest: images/test\nnames:\n%s\n' % (out.resolve(), train, names))
    stats.update({k: len(v) for k, v in parts.items()})
    with open(out / 'export_stats.json', 'w') as ofs:
        json.dump(stats, ofs, indent=1)
    return stats


# --------------------------------------------------------------------------------------------------------------
# Detections <-> challenge JSON (format of the professor's notebook, cell 23)
# --------------------------------------------------------------------------------------------------------------
def to_submission(dets, filenames):
    """dets: {filename: (boxes (N,4) xyxy, scores (N,), labels (N,))}; every filename gets an entry.

    Detections are written per image and per class sorted by descending confidence: the reference evaluator
    sorts the boxes but not their confidences (ISS-001), so pre-sorted input keeps them aligned."""
    out = {'images': {}, 'annotations': {}, 'categories': CATEGORIES}
    a_idx = 0
    for i, f in enumerate(filenames):
        boxes, scores, labels = dets.get(f, (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)))
        image_id = os.path.basename(f)
        out['images'][i] = {'image_id': image_id, 'filename': f, 'num_objects': int(len(boxes)), 'width': TILE, 'height': TILE}
        order = np.lexsort((-np.asarray(scores), np.asarray(labels)))
        for j in order:
            x1, y1, x2, y2 = boxes[j]
            out['annotations'][a_idx] = {'image_id': image_id, 'category_id': CATEGORIES[int(labels[j])],
                                         'bbox': (int(x1), int(y1), int(x2), int(y2)), 'confidence': str(float(scores[j]))}
            a_idx += 1
    return out


def save_dets(path, dets, filenames):
    """Compact storage of raw detections (npz) to re-evaluate / re-threshold without the GPU."""
    idx = {f: i for i, f in enumerate(filenames)}
    rows = [(idx[f], b, s, l) for f, (b, s, l) in dets.items() if len(b)]
    img = np.concatenate([np.full(len(b), i, np.int32) for i, b, _, _ in rows]) if rows else np.zeros(0, np.int32)
    np.savez_compressed(path, filenames=np.array(filenames), image=img,
                        boxes=np.concatenate([b for _, b, _, _ in rows]).astype(np.float32) if rows else np.zeros((0, 4), np.float32),
                        scores=np.concatenate([s for _, _, s, _ in rows]).astype(np.float32) if rows else np.zeros(0, np.float32),
                        labels=np.concatenate([l for _, _, _, l in rows]).astype(np.int8) if rows else np.zeros(0, np.int8))


def load_dets(path):
    z = np.load(path)
    filenames = z['filenames'].tolist()
    dets = {}
    for i, f in enumerate(filenames):
        m = z['image'] == i
        dets[f] = (z['boxes'][m].astype(np.float64), z['scores'][m].astype(np.float64), z['labels'][m].astype(np.int64))
    return dets, filenames


if __name__ == '__main__':
    anns = load_annotations()
    split = get_split(anns)
    print('split: train %d, val %d (%d escenas), desviación máx. %.4f' % (len(split['train']), len(split['val']),
                                                                      len(split['val_scenes']), split['max_share_deviation']))
    print('cajas:', split['counts'])
    print('export:', export_yolo(anns, split))
