"""torchvision single-stage detectors (SSD, SSDlite, RetinaNet) on xview_detection (CHG-005).

Same protocol as the YOLO runs: train split -> best epoch by val mAP50 (challenge metric) -> val detections.
torchvision detection models reserve label 0 for background, so class k of the challenge is label k + 1.
"""
import csv
import math
import os
import time

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models import detection as tvd
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection import ssd as tv_ssd
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

import det_metrics
from xview_det import CATEGORIES, DATA, clip_boxes, get_split, load_annotations

N_CLS = len(CATEGORIES)


class XviewDet(Dataset):
    def __init__(self, files, anns=None, train=False):
        self.files, self.anns, self.train = files, anns, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        img = cv2.imread(str(DATA / f), cv2.IMREAD_COLOR)[..., ::-1]          # RGB (checked == rasterio in gpu_check)
        h, w = img.shape[:2]
        target = {}
        if self.anns is not None:
            a = self.anns[f]
            boxes, labels, _ = clip_boxes(a['boxes'], a['labels'], w, h)
            boxes = boxes.astype(np.float32)
            if self.train:
                if np.random.rand() < 0.5:
                    img = img[:, ::-1]
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                if np.random.rand() < 0.5:
                    img = img[::-1]
                    boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
            target = {'boxes': torch.from_numpy(boxes.reshape(-1, 4)), 'labels': torch.from_numpy(labels + 1).long()}
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255)
        return x, target, f


def collate(batch):
    return tuple(zip(*batch))


def build(cfg):
    a, name = cfg['tv_args'], cfg['model']
    infer = cfg['infer']
    common = dict(score_thresh=infer['conf'], nms_thresh=infer.get('iou', 0.5), detections_per_img=infer['max_det'])
    if name == 'ssd300_vgg16':
        w = tvd.SSD300_VGG16_Weights.COCO_V1 if cfg['pretrained'] else None
        bw = None if cfg['pretrained'] else torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES
        model = tvd.ssd300_vgg16(weights=w, weights_backbone=bw, topk_candidates=4000, **common)
        if a.get('anchor_scales'):
            model.anchor_generator = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2]], scales=a['anchor_scales'])
        num_anchors = model.anchor_generator.num_anchors_per_location()
        in_ch = det_utils.retrieve_out_channels(model.backbone, (300, 300))
        model.head.classification_head = tv_ssd.SSDClassificationHead(in_ch, num_anchors, N_CLS + 1)
    elif name == 'ssdlite320':
        w = tvd.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1 if cfg['pretrained'] else None
        model = tvd.ssdlite320_mobilenet_v3_large(weights=w, topk_candidates=4000, **common)
        from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
        from functools import partial
        num_anchors = model.anchor_generator.num_anchors_per_location()
        in_ch = det_utils.retrieve_out_channels(model.backbone, (320, 320))
        norm = partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)
        model.head.classification_head = SSDLiteClassificationHead(in_ch, num_anchors, N_CLS + 1, norm)
    elif name == 'retinanet_r50_fpn_v2':
        w = tvd.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1 if cfg['pretrained'] else None
        bw = None if cfg['pretrained'] else torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        model = tvd.retinanet_resnet50_fpn_v2(weights=w, weights_backbone=bw, topk_candidates=2000, **common)
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(256, num_anchors, N_CLS + 1,
                                                                     norm_layer=lambda c: torch.nn.GroupNorm(32, c))
    else:
        raise ValueError(name)
    if a.get('min_size'):    # input resolution of GeneralizedRCNNTransform (SSD: fixed size; RetinaNet: min/max side)
        s = a['min_size']
        if name.startswith('ssd'):
            model.transform.fixed_size = (s, s)
            model.transform.min_size, model.transform.max_size = (s,), s
        else:
            model.transform.min_size, model.transform.max_size = (s,), max(s, 1333)
    return model


def _loader(files, anns, train, bs, workers):
    return DataLoader(XviewDet(files, anns, train), batch_size=bs, shuffle=train, num_workers=workers,
                      collate_fn=collate, pin_memory=True, persistent_workers=workers > 0, drop_last=train)


@torch.no_grad()
def _detect(model, loader, dev):
    model.eval()
    dets = {}
    for xs, _, fs in loader:
        with torch.autocast('cuda', dtype=torch.float16):
            out = model([x.to(dev, non_blocking=True) for x in xs])
        for f, o in zip(fs, out):
            dets[f] = (o['boxes'].float().cpu().numpy().astype(np.float64), o['scores'].float().cpu().numpy().astype(np.float64),
                       o['labels'].cpu().numpy().astype(np.int64) - 1)
    return dets


@torch.no_grad()
def _val_loss(model, loader, dev):
    model.train()       # torchvision detectors only return losses in train mode; no_grad -> no update
    bn_state = [(m, m.momentum) for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m, _ in bn_state:
        m.momentum = 0.0   # do not update BN running stats with val data
    tot, n = {}, 0
    for xs, ts, _ in loader:
        with torch.autocast('cuda', dtype=torch.float16):
            losses = model([x.to(dev) for x in xs], [{k: v.to(dev) for k, v in t.items()} for t in ts])
        for k, v in losses.items():
            tot[k] = tot.get(k, 0.0) + float(v) * len(xs)
        n += len(xs)
    for m, mom in bn_state:
        m.momentum = mom
    return {'val_' + k: v / n for k, v in tot.items()}


def train(run_id, cfg, run_dir):
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    run_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')
    a = cfg['tv_args']
    anns = load_annotations()
    split = get_split(anns)
    train_files = split['train'][:int(len(split['train']) * cfg['fraction'])]
    val_files = split['val'] if cfg['fraction'] >= 1 else split['val'][:max(16, int(len(split['val']) * cfg['fraction']))]
    tl = _loader(train_files, anns, True, cfg['batch'], a['workers'])
    vl = _loader(val_files, anns, False, cfg['batch'], a['workers'])
    model = build(cfg).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a['lr'], momentum=a['momentum'], weight_decay=a['weight_decay'])
    total = cfg['epochs'] * len(tl)
    warm = min(a['warmup_iters'], total // 4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: (0.001 + 0.999 * it / warm) if it < warm
                                              else 0.5 * (1 + math.cos(math.pi * (it - warm) / max(total - warm, 1))))
    scaler = torch.amp.GradScaler('cuda')
    gt = anns
    history, best, best_path = [], -1.0, run_dir / 'best.pt'
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        t0, sums, n = time.time(), {}, 0
        for xs, ts, _ in tl:
            xs = [x.to(dev, non_blocking=True) for x in xs]
            ts = [{k: v.to(dev, non_blocking=True) for k, v in t.items()} for t in ts]
            with torch.autocast('cuda', dtype=torch.float16):
                losses = model(xs, ts)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError('loss no finita en la época %d' % epoch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt); scaler.update(); sched.step()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            n += 1
        row = {'epoch': epoch, 'lr': opt.param_groups[0]['lr'], 'train_time_s': round(time.time() - t0, 1)}
        row.update({'train_' + k: v / n for k, v in sums.items()})
        row['train_loss'] = sum(v / n for v in sums.values())
        row.update(_val_loss(model, vl, dev))
        row['val_loss'] = sum(v for k, v in row.items() if k.startswith('val_') and k != 'val_loss')
        if epoch % a['eval_every'] == 0 or epoch == cfg['epochs']:
            m = det_metrics.evaluate(_detect(model, vl, dev), gt, val_files)
            row.update({'val_mAP50': m['mAP50'], 'val_mean_recall': m['mean_recall'], 'val_mean_precision': m['mean_precision']})
            if m['mAP50'] > best:
                best = m['mAP50']
                torch.save(model.state_dict(), best_path)
        history.append(row)
        print('epoch %d/%d %s' % (epoch, cfg['epochs'], {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        with open(run_dir / 'history.csv', 'w', newline='') as ofs:
            keys = sorted({k for r in history for k in r}, key=lambda k: (k != 'epoch', k))
            w = csv.DictWriter(ofs, fieldnames=keys)
            w.writeheader(); w.writerows(history)
    return best_path, history


def predict(weights, cfg, files):
    dev = torch.device('cuda')
    model = build(cfg)
    model.load_state_dict(torch.load(weights, map_location='cpu'))
    model.to(dev)
    return _detect(model, _loader(files, None, False, cfg['batch'], cfg['tv_args']['workers']), dev)
