"""Builds stick-figure sign animations (GIFs) from your recorded keyframes in data/processed/.
Animations are drawn MIRRORED so a learner can copy them exactly like the live (mirrored) camera view.
You can override any phrase with your own clip: put assets/signs/<phrase>.mp4 (or .gif / .webm)."""
import glob, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import config as C

ASSET_DIR = "assets/signs"
W, H, SCALE, OX, OY = 400, 440, 95, 200, 190
BG, BODY, FACE = (22, 25, 32), (205, 210, 220), (150, 155, 165)
LEFT, RIGHT = (80, 170, 255), (255, 160, 60)
LH0, RH0 = 25, 46
BODY_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
              (18, 19), (19, 20), (0, 17)]


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def _dense(B, up=4):
    """(K,67,2) keyframes -> smooth dense frames + visibility mask."""
    K = len(B)
    m = np.abs(B).sum(-1) > 0
    filled, t = B.copy(), np.arange(K)
    for p in range(B.shape[1]):
        v = np.where(m[:, p])[0]
        if len(v):
            for c in range(2):
                filled[:, p, c] = np.interp(t, v, B[v, p, c])
    tt = np.linspace(0, K - 1, (K - 1) * up + 1)
    i0 = np.floor(tt).astype(int).clip(0, K - 2)
    a = (tt - i0)[:, None, None]
    D = filled[i0] * (1 - a) + filled[i0 + 1] * a
    vis = m[i0] | m[i0 + 1]
    S = D.copy()
    S[1:-1] = (D[:-2] + 2 * D[1:-1] + D[2:]) / 4          # light smoothing
    return S, vis


def _frame(D, vis, label, font):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    pt = lambda i: (OX - D[i, 0] * SCALE, OY + D[i, 1] * SCALE)   # x flipped = mirror view

    def line(a, b, col, w):
        if vis[a] and vis[b]:
            d.line([pt(a), pt(b)], fill=col, width=w)

    for a, b in BODY_EDGES:
        line(a, b, BODY, 5)
    if vis[7] and vis[8]:
        hx, hy = (np.array(pt(7)) + np.array(pt(8))) / 2
    elif vis[0]:
        hx, hy = pt(0)
    else:
        hx = hy = None
    if hx is not None:
        r = 0.27 * SCALE
        d.ellipse([hx - r, hy - r, hx + r, hy + r], outline=FACE, width=3)
        for e in (2, 5):
            if vis[e]:
                x, y = pt(e); d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=FACE)
        line(9, 10, FACE, 3)
    line(15, LH0, LEFT, 3)
    line(16, RH0, RIGHT, 3)
    for base, col in ((LH0, LEFT), (RH0, RIGHT)):
        for a, b in HAND_EDGES:
            line(base + a, base + b, col, 3)
        for j in range(21):
            if vis[base + j]:
                x, y = pt(base + j); d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)
    d.text((14, 10), label, font=font, fill=(255, 255, 255))
    return im


def render_gif(skel, label, path, up=4, ms=45):
    D, vis = _dense(np.asarray(skel, np.float32), up)
    font = _font(24)
    frames = [_frame(D[i], vis[i], label, font) for i in range(len(D))]
    frames = [frames[0]] * 6 + frames + [frames[-1]] * 14          # short pause at start/end of loop
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0)
    return path


def build_for_phrase(phrase, force=False):
    """Pick the most 'typical' (medoid) recording of the phrase and render it. Returns path or None."""
    out = os.path.join(ASSET_DIR, f"{phrase}.gif")
    if os.path.exists(out) and not force:
        return out
    files = glob.glob(os.path.join(C.PROC_DIR, f"{phrase}__*.npz"))
    if not files:
        return None
    X = np.stack([np.load(f)["skel"] for f in files])
    best = int(np.argmin([np.abs(X - X[i]).mean() for i in range(len(X))])) if len(X) > 1 else 0
    return render_gif(X[best], phrase.replace("_", " ").upper(), out)


def get_animation(phrase):
    """Path to the media to show for a phrase (custom clip > generated GIF), or None."""
    for ext in ("mp4", "webm", "gif"):
        p = os.path.join(ASSET_DIR, f"{phrase}.{ext}")
        if os.path.exists(p):
            return p
    return build_for_phrase(phrase)
