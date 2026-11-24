"""Challenge metrics: mAP@50, Mean Accuracy, Mean Precision, Mean Recall (CHG-003).

`reference_evaluate` is a literal port of cells 16-19 of the professor's detector_example.ipynb (the metric the
Codabench leaderboard reports). `evaluate` reproduces it with the same matching rules but vectorised per image,
and keeps one record per detection so thresholds can be swept and images bootstrapped without re-matching.
`tests: python det_metrics.py` checks both implementations agree and that GT-as-prediction gives mAP = 1.

Matching rules of the reference (kept on purpose, they are what the leaderboard measures):
- per image and class, predictions in descending confidence; each one is compared with ALL GT boxes of the class
  and only the max-IoU GT counts: TP if IoU > 0.5 and that GT is still unmatched, otherwise FP (even if another
  unmatched GT overlaps > 0.5).
- IoU with the +1 px convention (VOC devkit), raw GT boxes (not clipped to the tile).
- AP: all-point interpolated area under the PR envelope (VOC 2010+), mAP = mean over the 4 classes.
- Mean Accuracy / Precision / Recall: sklearn macro averages over y_true / y_pred where an FP is
  (BACKGROUND -> cls) and an FN is (cls -> BACKGROUND), zero_division=1; BACKGROUND is one of the averaged labels.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

from xview_det import CATEGORIES

BACKGROUND = 'BACKGROUND'
IOU_THR = 0.5


# ---------------------------------------------------------------------------------------------------------------
# Reference implementation (professor's cells 16, 18, 19; only plotting removed, prints returned instead)
# ---------------------------------------------------------------------------------------------------------------
def area_intersection(boxes, box):
    xmin = np.maximum(np.min(boxes[:, 0::2], axis=1), np.min(box[0::2]))
    ymin = np.maximum(np.min(boxes[:, 1::2], axis=1), np.min(box[1::2]))
    xmax = np.minimum(np.max(boxes[:, 0::2], axis=1), np.max(box[0::2]))
    ymax = np.minimum(np.max(boxes[:, 1::2], axis=1), np.max(box[1::2]))
    w = np.maximum(xmax - xmin + 1.0, 0.0)
    h = np.maximum(ymax - ymin + 1.0, 0.0)
    return w * h


def area_union(boxes, box):
    area_anns = (np.max(box[0::2]) - np.min(box[0::2]) + 1.0) * (np.max(box[1::2]) - np.min(box[1::2]) + 1.0)
    area_pred = (np.max(boxes[:, 0::2], axis=1) - np.min(boxes[:, 0::2], axis=1) + 1.0) * (np.max(boxes[:, 1::2], axis=1) - np.min(boxes[:, 1::2], axis=1) + 1.0)
    return area_anns + area_pred - area_intersection(boxes, box)


def calc_iou(boxes, box):
    iou = area_intersection(boxes, box) / area_union(boxes, box)
    return np.max(iou), np.argmax(iou)


def calc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


def reference_evaluate(predictions, annotations, categories=CATEGORIES, threshold=IOU_THR):
    """predictions: {f: {cls_name: {'bbox': [...], 'confidence': [...]}}}; annotations: {f: {cls_name: {'bbox': [...]}}}."""
    y_true, y_pred = [], []
    tps, confidences = dict(), dict()
    for cls in categories.values():
        tps[cls], confidences[cls] = [], []
        for f in predictions:
            pred_boxes, pred_confidences = [], []
            if cls in predictions[f].keys():
                for idx in range(len(predictions[f][cls]['bbox'])):
                    pred_boxes.append(predictions[f][cls]['bbox'][idx])
                    pred_confidences.append(predictions[f][cls]['confidence'][idx])
                sorted_ind = np.argsort(-np.array(pred_confidences))
                pred_boxes = np.array(pred_boxes)[sorted_ind, :]
            pred_boxes = np.array(pred_boxes).astype(float)
            anno_boxes = []
            if cls in annotations[f].keys():
                anno_boxes = annotations[f][cls]['bbox']
            anno_boxes = np.array(anno_boxes).astype(float)
            anno_indices = list(range(len(anno_boxes)))
            for pred_idx, pred_box in enumerate(pred_boxes):
                iou_value, ann_index = calc_iou(anno_boxes, pred_box) if len(anno_boxes) > 0 else (-1, -1)
                if iou_value > threshold and ann_index in anno_indices:
                    anno_indices.remove(int(ann_index))
                    tps[cls] += [1.0]
                    y_true += [cls]
                else:
                    tps[cls] += [0.0]
                    y_true += [BACKGROUND]
                y_pred += [cls]
                confidences[cls] += [pred_confidences[pred_idx]]   # ISS-001: index of the UNsorted list
            y_true += [cls] * len(anno_indices)
            y_pred += [BACKGROUND] * len(anno_indices)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ap_list = {}
    for cls in categories.values():
        sorted_ind = np.argsort(-np.array(confidences[cls]))
        tp = np.cumsum(np.array(tps[cls])[sorted_ind], dtype=float)
        recall = np.array([0.0]) if len(tp) == 0 else tp / np.maximum(np.sum(y_true == cls), np.finfo(np.float64).eps)
        precision = np.array([0.0]) if len(tp) == 0 else tp / np.maximum(list(range(1, np.sum(y_pred == cls) + 1)), np.finfo(np.float64).eps)
        ap_list[cls] = calc_ap(recall, precision)
    return {'mAP50': float(np.mean(list(ap_list.values()))),
            'mean_accuracy': accuracy_score(y_true, y_pred),
            'mean_recall': recall_score(y_true, y_pred, average='macro', zero_division=1),
            'mean_precision': precision_score(y_true, y_pred, average='macro', zero_division=1),
            'AP': ap_list}


# ---------------------------------------------------------------------------------------------------------------
# Fast implementation with per-detection records
# ---------------------------------------------------------------------------------------------------------------
def _iou_matrix(pred, gt):
    """IoU (P, G) with the reference's +1 px convention and min/max coordinates."""
    px1, px2 = np.minimum(pred[:, 0], pred[:, 2]), np.maximum(pred[:, 0], pred[:, 2])
    py1, py2 = np.minimum(pred[:, 1], pred[:, 3]), np.maximum(pred[:, 1], pred[:, 3])
    gx1, gx2 = np.minimum(gt[:, 0], gt[:, 2]), np.maximum(gt[:, 0], gt[:, 2])
    gy1, gy2 = np.minimum(gt[:, 1], gt[:, 3]), np.maximum(gt[:, 1], gt[:, 3])
    w = np.maximum(np.minimum(px2[:, None], gx2[None]) - np.maximum(px1[:, None], gx1[None]) + 1.0, 0.0)
    h = np.maximum(np.minimum(py2[:, None], gy2[None]) - np.maximum(py1[:, None], gy1[None]) + 1.0, 0.0)
    inter = w * h
    ap = (px2 - px1 + 1.0) * (py2 - py1 + 1.0)
    ag = (gx2 - gx1 + 1.0) * (gy2 - gy1 + 1.0)
    return inter / (ap[:, None] + ag[None] - inter)


class Records:
    """One row per detection (image, class, confidence, tp) + GT count per (image, class)."""

    def __init__(self, image, cls, conf, tp, n_gt):
        self.image, self.cls, self.conf, self.tp, self.n_gt = image, cls, conf, tp, n_gt   # n_gt: (n_images, n_cls)

    def filter(self, min_conf=0.0, max_det=None):
        keep = self.conf >= min_conf
        if max_det is not None:   # top-k per image over all classes, as a detector's max_det would
            order = np.lexsort((-self.conf, self.image))
            rank = np.empty(len(order), int)
            starts = np.r_[0, np.flatnonzero(np.diff(self.image[order])) + 1]
            counts = np.diff(np.r_[starts, len(order)])
            rank[order] = np.arange(len(order)) - np.repeat(starts, counts)
            keep &= rank < max_det
        return Records(self.image[keep], self.cls[keep], self.conf[keep], self.tp[keep], self.n_gt)

    def subset(self, images):
        """Bootstrap: rebuild the records for a list of image indices (with repetition)."""
        order = np.argsort(self.image, kind='stable')
        starts = np.searchsorted(self.image[order], np.arange(len(self.n_gt)))
        ends = np.searchsorted(self.image[order], np.arange(len(self.n_gt)), side='right')
        idx = np.concatenate([order[starts[i]:ends[i]] for i in images]) if len(images) else np.zeros(0, int)
        new_img = np.concatenate([np.full(ends[i] - starts[i], k) for k, i in enumerate(images)]) if len(images) else np.zeros(0, int)
        return Records(new_img, self.cls[idx], self.conf[idx], self.tp[idx], self.n_gt[images])


def match(dets, gt, filenames, n_cls=len(CATEGORIES), thr=IOU_THR):
    """dets: {f: (boxes, scores, labels)}; gt: load_annotations() dict. Returns Records over `filenames`."""
    img_l, cls_l, conf_l, tp_l = [], [], [], []
    n_gt = np.zeros((len(filenames), n_cls), np.int64)
    for i, f in enumerate(filenames):
        g = gt[f]
        pb, ps, pl = dets.get(f, (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)))
        for c in range(n_cls):
            gb = g['boxes'][g['labels'] == c]
            n_gt[i, c] = len(gb)
            m = pl == c
            if not m.any():
                continue
            order = np.argsort(-ps[m])                      # same (quicksort) order as the reference
            boxes, scores = pb[m][order], ps[m][order]
            tp = np.zeros(len(boxes))
            if len(gb):
                iou = _iou_matrix(boxes.astype(float), gb)
                best, arg = iou.max(1), iou.argmax(1)
                used = np.zeros(len(gb), bool)
                for k in np.flatnonzero(best > thr):         # greedy in confidence order
                    if not used[arg[k]]:
                        used[arg[k]] = True
                        tp[k] = 1.0
            img_l.append(np.full(len(boxes), i)); cls_l.append(np.full(len(boxes), c))
            conf_l.append(scores); tp_l.append(tp)
    cat = lambda l, dt: np.concatenate(l).astype(dt) if l else np.zeros(0, dt)   # noqa: E731
    return Records(cat(img_l, np.int64), cat(cls_l, np.int64), cat(conf_l, np.float64), cat(tp_l, np.float64), n_gt)


def pr_curve(rec, c):
    m = rec.cls == c
    order = np.argsort(-rec.conf[m])
    tp = np.cumsum(rec.tp[m][order])
    n = rec.n_gt[:, c].sum()
    if len(tp) == 0:
        return np.array([0.0]), np.array([0.0]), rec.conf[m][order]
    recall = tp / max(n, np.finfo(np.float64).eps)
    precision = tp / np.arange(1, len(tp) + 1)
    return recall, precision, rec.conf[m][order]


def metrics(rec, n_cls=len(CATEGORIES)):
    """The 4 leaderboard metrics + per-class AP / recall / precision + detection confusion counts."""
    out, aps = {}, []
    tp_c = np.array([rec.tp[rec.cls == c].sum() for c in range(n_cls)])
    det_c = np.array([(rec.cls == c).sum() for c in range(n_cls)])
    gt_c = rec.n_gt.sum(0)
    for c in range(n_cls):
        r, p, _ = pr_curve(rec, c)
        ap = calc_ap(r, p)
        aps.append(ap)
        name = CATEGORIES[c]
        out['AP_' + name] = float(ap)
        out['recall_' + name] = float(tp_c[c] / max(gt_c[c], 1))
        out['precision_' + name] = float(tp_c[c] / det_c[c]) if det_c[c] else 1.0
    # y_true / y_pred exactly as the reference builds them (TP: c->c, FP: BG->c, FN: c->BG), as int codes (BG = -1)
    fp_c, fn_c = det_c - tp_c, gt_c - tp_c
    y_true = np.concatenate([np.repeat(np.arange(n_cls), tp_c.astype(int)), np.full(int(fp_c.sum()), -1), np.repeat(np.arange(n_cls), fn_c.astype(int))])
    y_pred = np.concatenate([np.repeat(np.arange(n_cls), tp_c.astype(int)), np.repeat(np.arange(n_cls), fp_c.astype(int)), np.full(int(fn_c.sum()), -1)])
    out.update({'mAP50': float(np.mean(aps)),
                'mean_accuracy': float(accuracy_score(y_true, y_pred)),
                'mean_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=1)),
                'mean_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=1)),
                'n_det': int(det_c.sum()), 'n_gt': int(gt_c.sum())})
    out['confusion'] = confusion(tp_c, fp_c, fn_c)
    return out


def confusion(tp_c, fp_c, fn_c):
    """5x5 matrix like the reference's sklearn confusion_matrix, labels [BACKGROUND, *classes] (rows = annotation)."""
    n = len(tp_c)
    cm = np.zeros((n + 1, n + 1), np.int64)
    for c in range(n):
        cm[c + 1, c + 1] = tp_c[c]
        cm[0, c + 1] = fp_c[c]
        cm[c + 1, 0] = fn_c[c]
    return cm.tolist()


def evaluate(dets, gt, filenames, min_conf=0.0, max_det=None):
    return metrics(match(dets, gt, filenames).filter(min_conf, max_det))


def bootstrap_delta(rec_a, rec_b, n_boot=500, seed=0, key='mAP50'):
    """Paired bootstrap over images of metric(b) - metric(a); both records must cover the same filenames."""
    assert rec_a.n_gt.shape == rec_b.n_gt.shape
    rng = np.random.default_rng(seed)
    n = len(rec_a.n_gt)
    deltas = []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        deltas.append(_quick(rec_b.subset(s), key) - _quick(rec_a.subset(s), key))
    deltas = np.array(deltas)
    return {'delta': _quick(rec_b, key) - _quick(rec_a, key), 'ci95': np.percentile(deltas, [2.5, 97.5]).tolist(),
            'p_le_0': float((deltas <= 0).mean()), 'n_boot': n_boot}


def _quick(rec, key):
    if key == 'mAP50':
        return float(np.mean([calc_ap(*pr_curve(rec, c)[:2]) for c in range(rec.n_gt.shape[1])]))
    return metrics(rec)[key]


# ---------------------------------------------------------------------------------------------------------------
def _to_reference(dets, gt, filenames):
    preds, anns = {}, {}
    for f in filenames:
        preds[f], anns[f] = {}, {}
        for b, c in zip(gt[f]['boxes'], gt[f]['labels']):
            anns[f].setdefault(CATEGORIES[int(c)], {'bbox': []})['bbox'].append(tuple(b))
        pb, ps, pl = dets.get(f, (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)))
        for b, s, c in zip(pb, ps, pl):
            d = preds[f].setdefault(CATEGORIES[int(c)], {'bbox': [], 'confidence': []})
            d['bbox'].append(tuple(b)); d['confidence'].append(s)
    return preds, anns


def _self_test(n_images=60, seed=0):
    from xview_det import get_split, load_annotations
    gt = load_annotations()
    files = get_split(gt)['val'][:n_images]
    rng = np.random.default_rng(seed)
    # 1) GT as prediction -> mAP = 1
    dets = {f: (gt[f]['boxes'], np.ones(len(gt[f]['boxes'])), gt[f]['labels']) for f in files}
    m = evaluate(dets, gt, files)
    assert abs(m['mAP50'] - 1.0) < 1e-9, m['mAP50']
    print('GT como predicción: mAP50 = %.4f, mAcc = %.4f, mPrec = %.4f, mRec = %.4f'
          % (m['mAP50'], m['mean_accuracy'], m['mean_precision'], m['mean_recall']))
    # 2) noisy predictions, sorted by confidence (as to_submission writes them): fast == reference
    dets = {}
    for f in files:
        b, l = gt[f]['boxes'], gt[f]['labels']
        keep = rng.random(len(b)) < 0.7
        nb = b[keep] + rng.normal(0, 4, (keep.sum(), 4))
        fake = rng.uniform(0, 600, (len(b) // 3, 4)); fake[:, 2:] = fake[:, :2] + rng.uniform(5, 40, (len(fake), 2))
        boxes = np.vstack([nb, fake])
        labels = np.r_[np.where(rng.random(keep.sum()) < 0.9, l[keep], rng.integers(0, 4, keep.sum())), rng.integers(0, 4, len(fake))]
        scores = rng.random(len(boxes))
        o = np.lexsort((-scores, labels))
        dets[f] = (boxes[o], scores[o], labels[o])
    fast = evaluate(dets, gt, files)
    ref = reference_evaluate(*_to_reference(dets, gt, files))
    for k in ['mAP50', 'mean_accuracy', 'mean_precision', 'mean_recall']:
        print('%-15s fast %.6f  reference %.6f' % (k, fast[k], ref[k]))
        assert abs(fast[k] - ref[k]) < 1e-9, k
    print('OK: implementación rápida == referencia del profesor')


if __name__ == '__main__':
    _self_test()
