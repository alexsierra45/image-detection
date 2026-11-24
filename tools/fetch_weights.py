"""Download pretrained weights and assets on the LOGIN node (download only, no compute), so jobs run offline (CHG-001).

    source slurm/env.sh && .venv/bin/python tools/fetch_weights.py

- Ultralytics: yolo11{n,s,m}.pt -> cache/weights/ ; Arial.ttf (plot labels) -> $YOLO_CONFIG_DIR
- torchvision COCO detectors -> $TORCH_HOME/hub/checkpoints/
"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / 'cache' / 'weights'
for var in ('YOLO_CONFIG_DIR', 'TORCH_HOME'):
    assert os.environ.get(var), 'falta %s: ejecutar antes `source slurm/env.sh`' % var
os.environ.pop('YOLO_OFFLINE', None)
WEIGHTS.mkdir(parents=True, exist_ok=True)

YOLO = sys.argv[1:] or ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt']

from ultralytics.utils.checks import check_font  # noqa: E402
from ultralytics.utils.downloads import attempt_download_asset  # noqa: E402

for name in YOLO:
    dst = WEIGHTS / name
    if not dst.exists():
        got = Path(attempt_download_asset(name))
        shutil.move(str(got), dst)
    print('ok', dst, '%.1f MB' % (dst.stat().st_size / 2**20))
print('font', check_font('Arial.ttf'))

import torchvision.models.detection as tvd  # noqa: E402

for builder, weights in [(tvd.ssd300_vgg16, tvd.SSD300_VGG16_Weights.COCO_V1),
                         (tvd.ssdlite320_mobilenet_v3_large, tvd.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1),
                         (tvd.retinanet_resnet50_fpn_v2, tvd.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)]:
    weights.get_state_dict(progress=False)   # downloads to $TORCH_HOME/hub/checkpoints
    print('ok', weights)
