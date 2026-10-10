"""Build a defect-filtered dataset from diagnose_artifacts.py scores."""

import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description='Copy artifact-free generated images into a clean dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--scores', required=True,
                    help='scores_gen.json from diagnose_artifacts.py')
    ap.add_argument('--images', required=True, help='Generated images dir')
    ap.add_argument('--out-dir', required=True, help='Output dir for clean set')
    ap.add_argument('--keep', type=int, default=10000,
                    help='Max images to keep (0 = all clean images)')
    ap.add_argument('--max-flags', type=int, default=0,
                    help='Keep images with at most this many flags')
    ap.add_argument('--min-sharpness', type=float, default=None,
                    help='Additionally drop images below this sharpness')
    ap.add_argument('--min-rim-contrast', type=float, default=None,
                    help='Additionally drop images below this rim contrast')
    args = ap.parse_args()

    with open(args.scores) as f:
        scores = json.load(f)
    img_dir = Path(args.images)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kept, dropped = [], 0
    for name, s in sorted(scores.items()):
        ok = s['n_flags'] <= args.max_flags
        if ok and args.min_sharpness is not None:
            ok = s['sharpness'] >= args.min_sharpness
        if ok and args.min_rim_contrast is not None:
            ok = s['rim_contrast'] >= args.min_rim_contrast
        if ok:
            kept.append(name)
        else:
            dropped += 1

    if args.keep > 0 and len(kept) > args.keep:
        kept = kept[:args.keep]

    name_to_path = {}
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        for p in img_dir.rglob(ext):
            name_to_path.setdefault(p.name, p)

    n_copied = 0
    for name in kept:
        src = name_to_path.get(name)
        if src is not None:
            shutil.copy(src, out_dir / name)
            n_copied += 1

    print(f'Scored: {len(scores)}  dropped: {dropped} '
          f'({100 * dropped / len(scores):.1f}%)  copied: {n_copied}')
    print(f'Clean set: {out_dir}')
    if n_copied < args.keep:
        print(f'NOTE: only {n_copied} clean images available for '
              f'--keep {args.keep}; generate a larger surplus.')


if __name__ == '__main__':
    main()
