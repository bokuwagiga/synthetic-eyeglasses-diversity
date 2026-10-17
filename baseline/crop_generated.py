"""Crop 512x512 StyleGAN2-ADA outputs back to the 256x512 content band.

The baseline was trained on images padded with 128 rows above and below the
frame (see prepare_data.py), so rows 128-383 of every generated image hold
the frame. The crops are written with the same file names and can then be
evaluated with the scripts in metrics/ exactly like our own models.

  python baseline/crop_generated.py --src ../sg2ada_gen/snap001800 \
      --dst results/baseline/generated/images
"""
import argparse
import os

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='folder with 512x512 PNGs from gen_images.py')
    ap.add_argument('--dst', required=True, help='output folder for the 256x512 crops')
    ap.add_argument('--pad', type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    files = sorted(f for f in os.listdir(args.src) if f.lower().endswith('.png'))
    for f in files:
        im = Image.open(os.path.join(args.src, f)).convert('RGB')
        w, h = im.size
        im.crop((0, args.pad, w, h - args.pad)).save(os.path.join(args.dst, f))
    print('cropped %d images into %s' % (len(files), args.dst))


if __name__ == '__main__':
    main()
