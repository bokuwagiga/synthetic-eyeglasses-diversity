"""Pad the 256x512 catalogue images to 512x512 for the StyleGAN2-ADA baseline.

The official StyleGAN2-ADA code trains on square images only. Each image is
centre-cropped to 2:1 and resized to 256x512 exactly as in
metrics/evaluate_diversity.py, then padded with 128 rows above and 128 rows
below, filled with the image's own background colour (the median of its
border pixels). The padded images are written as PNG; convert them to a
StyleGAN dataset zip with the official dataset_tool.py afterwards:

  python baseline/prepare_data.py --images data/source/images --out ../sg2ada_data/padded
  python <stylegan3>/dataset_tool.py --source ../sg2ada_data/padded \
      --dest ../sg2ada_data/catalogue_512.zip --resolution 512x512
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.evaluate_diversity import list_images, load_rgb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='folder of catalogue images')
    ap.add_argument('--out', required=True, help='output folder for the padded PNGs')
    ap.add_argument('--pad', type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    paths = list_images(args.images)
    for p in paths:
        img = load_rgb(p)
        h, w = img.shape[:2]
        border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).astype(np.float32)
        bg = np.clip(np.round(np.median(border, axis=0)), 0, 255).astype(np.uint8)
        canvas = np.empty((h + 2 * args.pad, w, 3), np.uint8)
        canvas[:] = bg
        canvas[args.pad:args.pad + h] = img
        name = os.path.splitext(os.path.basename(p))[0] + '.png'
        Image.fromarray(canvas).save(os.path.join(args.out, name))
    print('wrote %d padded images to %s' % (len(paths), args.out))


if __name__ == '__main__':
    main()
