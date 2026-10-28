"""Turn videos in data/raw/<phrase>/<signer>_<n>.mp4 into keyframe features (data/processed/*.npz)."""
import argparse, os, glob
import numpy as np
from tqdm import tqdm
import config as C
from signlib import extract_video, build_sample, RGBEncoder

ap = argparse.ArgumentParser()
ap.add_argument("--no-rgb", action="store_true", help="skeleton only (faster)")
a = ap.parse_args()
os.makedirs(C.PROC_DIR, exist_ok=True)
enc = None if a.no_rgb else RGBEncoder()
files = [f for ext in ("mp4", "mov", "avi", "webm", "mkv") for f in glob.glob(f"{C.RAW_DIR}/*/*.{ext}")]
print(f"{len(files)} videos found")
skipped = 0
for f in tqdm(sorted(files)):
    phrase = os.path.basename(os.path.dirname(f))
    stem = os.path.splitext(os.path.basename(f))[0]
    signer = stem.split("_")[0]
    out = os.path.join(C.PROC_DIR, f"{phrase}__{stem}.npz")
    if os.path.exists(out):
        continue
    P, M, crops = extract_video(f)
    res = build_sample(P, M, crops, enc) if len(P) else None
    if res is None:
        skipped += 1; print("skip (no body found):", f); continue
    skel, rgb = res
    kw = dict(skel=skel, phrase=phrase, signer=signer)
    if rgb is not None:
        kw["rgb"] = rgb
    np.savez_compressed(out, **kw)
print(f"done. skipped {skipped}")
