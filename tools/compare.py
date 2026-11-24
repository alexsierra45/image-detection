"""Paired bootstrap over val images between two runs (DEC-003, CHG-007). CPU only, reads experiments/preds/*.npz.

    python tools/compare.py <base_run_id> <new_run_id> [--n-boot 500] [--conf 0.001] [--max-det 1500]

Prints the 4 challenge metrics of both runs, delta mAP50 with a 95 % CI and P(delta <= 0), and writes
experiments/compare/<base>__<new>.json. Decision rule: improvement if the CI excludes 0.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import det_metrics  # noqa: E402
from train_det import EXP_DIR, PREDS_DIR  # noqa: E402
from xview_det import load_annotations, load_dets  # noqa: E402

KEYS = ['mAP50', 'mean_accuracy', 'mean_precision', 'mean_recall']


def records(run_id, gt, conf, max_det):
    dets, files = load_dets(PREDS_DIR / (run_id + '_val.npz'))
    return det_metrics.match(dets, gt, files).filter(conf, max_det), files


def main():
    p = argparse.ArgumentParser()
    p.add_argument('base'); p.add_argument('new')
    p.add_argument('--n-boot', type=int, default=500)
    p.add_argument('--conf', type=float, default=0.0)
    p.add_argument('--max-det', type=int)
    a = p.parse_args()
    gt = load_annotations()
    ra, fa = records(a.base, gt, a.conf, a.max_det)
    rb, fb = records(a.new, gt, a.conf, a.max_det)
    assert fa == fb, 'los dos runs no cubren las mismas imágenes de val'
    ma, mb = det_metrics.metrics(ra), det_metrics.metrics(rb)
    for k in KEYS:
        print('%-15s %s %.4f | %s %.4f | delta %+.4f' % (k, a.base, ma[k], a.new, mb[k], mb[k] - ma[k]))
    bs = det_metrics.bootstrap_delta(ra, rb, n_boot=a.n_boot)
    verdict = 'mejora' if bs['ci95'][0] > 0 else ('empeora' if bs['ci95'][1] < 0 else 'no concluyente')
    print('delta mAP50 %+.4f, IC95 [%+.4f, %+.4f], P(delta<=0) = %.3f -> %s'
          % (bs['delta'], bs['ci95'][0], bs['ci95'][1], bs['p_le_0'], verdict))
    out = EXP_DIR / 'compare'
    out.mkdir(exist_ok=True)
    with open(out / ('%s__%s.json' % (a.base, a.new)), 'w') as ofs:
        json.dump({'base': a.base, 'new': a.new, 'conf': a.conf, 'max_det': a.max_det,
                   'base_metrics': {k: ma[k] for k in KEYS}, 'new_metrics': {k: mb[k] for k in KEYS},
                   'bootstrap': bs, 'verdict': verdict}, ofs, indent=1)


if __name__ == '__main__':
    main()
