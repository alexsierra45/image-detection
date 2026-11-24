"""Challenge submission from a finished run (CHG-004): test detections -> submissions/<EXP>/prediction.{json,zip}.

    python predict.py <run_id> [--conf 0.001] [--max-det 1500] [--iou 0.7] [--augment] [--tag <name>]

Reads the run config from experiments/history/<run_id>.json, predicts the 852 test tiles, writes the JSON in
the professor's format (xview_det.to_submission), zips it and checks the schema. Needs a GPU (SLURM job).
"""
import argparse
import json
import zipfile

from train_det import HIST_DIR, PREDS_DIR, ROOT, predict_yolo
from xview_det import save_dets, test_filenames, to_submission


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_id')
    p.add_argument('--conf', type=float)
    p.add_argument('--max-det', type=int)
    p.add_argument('--iou', type=float)
    p.add_argument('--augment', action='store_true')
    p.add_argument('--tag', default='')
    a = p.parse_args()
    with open(HIST_DIR / (a.run_id + '.json')) as ifs:
        h = json.load(ifs)
    cfg, weights = h['config'], h['weights']
    infer = dict(cfg['infer'])
    for k, v in [('conf', a.conf), ('max_det', a.max_det), ('iou', a.iou)]:
        if v is not None:
            infer[k] = v
    if a.augment:
        infer['augment'] = True
    files = test_filenames()
    assert len(files) == 852, len(files)
    if cfg['family'] == 'yolo':
        dets = predict_yolo(weights, files, infer, cfg['imgsz'])
    else:
        import tv_detectors
        dets = tv_detectors.predict(weights, dict(cfg, infer=infer), files)
    save_dets(PREDS_DIR / (a.run_id + (a.tag and '_' + a.tag) + '_test.npz'), dets, files)
    sub = to_submission(dets, files)
    out = ROOT / 'submissions' / (h['row']['exp_id'] + (a.tag and '_' + a.tag))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'prediction.json', 'w') as ofs:
        json.dump(sub, ofs)
    with zipfile.ZipFile(out / 'prediction.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(out / 'prediction.json', 'prediction.json')
    with open(out / 'submission_info.json', 'w') as ofs:
        json.dump({'run_id': a.run_id, 'weights': weights, 'infer': infer, 'n_images': len(sub['images']),
                   'n_detections': len(sub['annotations'])}, ofs, indent=1)
    # schema check against the professor's notebook (cell 23)
    assert set(sub) == {'images', 'annotations', 'categories'}
    img0 = next(iter(sub['images'].values()))
    assert set(img0) == {'image_id', 'filename', 'num_objects', 'width', 'height'} and img0['filename'].startswith('xview_test/')
    if sub['annotations']:
        a0 = next(iter(sub['annotations'].values()))
        assert set(a0) == {'image_id', 'category_id', 'bbox', 'confidence'} and isinstance(a0['confidence'], str)
    print('OK %s: %d imágenes, %d detecciones -> %s' % (a.run_id, len(sub['images']), len(sub['annotations']), out / 'prediction.zip'))


if __name__ == '__main__':
    main()
