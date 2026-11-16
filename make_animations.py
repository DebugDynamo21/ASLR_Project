"""Generate sign animations for every phrase:  python make_animations.py   (run after extract_features.py)"""
import argparse, glob, os
import config as C
from animations import build_for_phrase

ap = argparse.ArgumentParser()
ap.add_argument("--phrases", nargs="*", default=None, help="default: every phrase found in data/processed")
a = ap.parse_args()
found = sorted({os.path.basename(f).split("__")[0] for f in glob.glob(f"{C.PROC_DIR}/*.npz")})
for p in (a.phrases or found):
    out = build_for_phrase(p, force=True)
    print(("made  " if out else "no data for ") + p, out or "")
