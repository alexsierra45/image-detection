"""GPU and image-reading sanity check, run inside slurm/gpu_check.sbatch (CHG-001)."""
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch
import torchvision

sys.path.insert(0, '.')
from xview_det import load_geoimage  # noqa: E402

print('python', sys.version.split()[0], '| torch', torch.__version__, '| torchvision', torchvision.__version__,
      '| cuda runtime', torch.version.cuda, '| cudnn', torch.backends.cudnn.version())
assert torch.cuda.is_available(), 'torch no ve ninguna GPU'
print('GPU:', torch.cuda.get_device_name(0), '| capability', torch.cuda.get_device_capability(0))
x = torch.randn(16, 3, 640, 640, device='cuda')
conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
torch.cuda.synchronize(); t = time.time()
for _ in range(20):
    y = conv(x)
torch.cuda.synchronize()
print('20 conv 3x3 (16x3x640x640 -> 64): %.3f s' % (time.time() - t))
with torch.autocast('cuda', dtype=torch.float16):
    y = conv(x)
print('autocast fp16 ok:', y.dtype)
boxes = torch.rand(2000, 4, device='cuda') * 600
boxes[:, 2:] += boxes[:, :2]
keep = torchvision.ops.nms(boxes, torch.rand(2000, device='cuda'), 0.5)
print('torchvision.ops.nms en GPU ok: %d cajas' % len(keep))

# OpenCV (lo que usa Ultralytics) frente a rasterio (lo que usa el notebook del profesor)
files = sorted(os.path.relpath(p, 'xview_detection') for p in glob.glob('xview_detection/xview_train/*.tif'))[:20]
for f in files:
    ref = load_geoimage(f)                       # RGB, HxWx3
    bgr = cv2.imread('xview_detection/' + f, cv2.IMREAD_COLOR)   # BGR
    assert bgr is not None and bgr.shape == ref.shape, (f, None if bgr is None else bgr.shape, ref.shape)
    assert np.array_equal(bgr[..., ::-1], ref), 'cv2 y rasterio difieren en ' + f
print('cv2.imread == rasterio (RGB<->BGR) en %d tif' % len(files))
import ultralytics  # noqa: E402
print('ultralytics', ultralytics.__version__)
print('OK')
