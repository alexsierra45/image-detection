"""EXP-008 / IDEA-001: the 4 challenge metrics vs export confidence threshold and max_det, from saved val detections.

    python tools/infer_sweep.py <run_id>

No GPU and no re-matching (det_metrics.Records.filter): greedy matching in confidence order makes the
matches of the surviving detections independent of the dropped low-confidence tail. Writes
experiments/infer/<run_id>_sweep.csv and a PNG with the 4 metrics vs conf.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import det_metrics  # noqa: E402
from train_det import EXP_DIR, PREDS_DIR  # noqa: E402
from xview_det import load_annotations, load_dets  # noqa: E402

CONFS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
MAX_DETS = [300, 1000, 1500]
KEYS = ['mAP50', 'mean_accuracy', 'mean_precision', 'mean_recall']


def main():
    run_id = sys.argv[1]
    gt = load_annotations()
    dets, files = load_dets(PREDS_DIR / (run_id + '_val.npz'))
    rec = det_metrics.match(dets, gt, files)
    rows = []
    for md in MAX_DETS:
        for c in CONFS:
            m = det_metrics.metrics(rec.filter(c, md))
            rows.append({'conf': c, 'max_det': md, 'n_det': m['n_det'], **{k: round(m[k], 5) for k in KEYS}})
            print(rows[-1], flush=True)
    out = EXP_DIR / 'infer'
    out.mkdir(exist_ok=True)
    with open(out / (run_id + '_sweep.csv'), 'w', newline='') as ofs:
        w = csv.DictWriter(ofs, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_theme import PALETTE, RC
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    md = max(MAX_DETS)
    sel = [r for r in rows if r['max_det'] == md]
    for k, col in zip(KEYS, PALETTE):
        ax.plot([r['conf'] for r in sel], [r[k] for r in sel], marker='o', color=col, label=k)
    ax.set_xscale('log'); ax.set_xlabel('export confidence threshold'); ax.set_ylabel('metric (val)')
    ax.set_title('%s: challenge metrics vs confidence (max_det %d)' % (run_id, md)); ax.legend()
    fig.savefig(out / (run_id + '_sweep.png'))
    print('->', out / (run_id + '_sweep.csv'))


if __name__ == '__main__':
    main()
