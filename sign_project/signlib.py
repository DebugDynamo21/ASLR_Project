"""Shared logic used by extraction, training and the app:
landmark extraction -> trimming -> keyframe selection -> features -> inference."""
from collections import deque
import cv2
import numpy as np
import config as C

N_POSE, N_HAND = 25, 21                 # upper-body pose points + 21 per hand
N_PTS = N_POSE + 2 * N_HAND             # 67 skeletal points
LH, RH = slice(25, 46), slice(46, 67)
L_SH, R_SH = 11, 12
FEAT_DIM = N_PTS * 2 + 2 * N_HAND * 2 + N_PTS * 2   # body + hand-local + velocity = 352


def resize_w(img, max_w=C.MAX_W):
    h, w = img.shape[:2]
    return cv2.resize(img, (max_w, int(h * max_w / w))) if w > max_w else img


# ----------------------------------------------------------------- landmarks
class LandmarkExtractor:
    def __init__(self, complexity=C.MP_COMPLEXITY):
        import mediapipe as mp
        self.mp = mp
        self.holo = mp.solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=complexity,
            smooth_landmarks=True, min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        res = self.holo.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        P = np.zeros((N_PTS, 2), np.float32)
        M = np.zeros(N_PTS, bool)

        def fill(lms, sl, n):
            if lms is None:
                return
            P[sl] = np.array([[l.x * w, l.y * h] for l in lms.landmark[:n]], np.float32)
            M[sl] = True

        fill(res.pose_landmarks, slice(0, N_POSE), N_POSE)
        fill(res.left_hand_landmarks, LH, N_HAND)
        fill(res.right_hand_landmarks, RH, N_HAND)
        return P, M, res

    def close(self):
        self.holo.close()


def crop_roi(frame, P, M, size=C.IMG_SIZE):
    """Square crop around the signer's upper body/hands -> RGB keyframe image."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = 0, 0, w, h
    if M.any():
        lo, hi = P[M].min(0), P[M].max(0)
        cx, cy = (lo + hi) / 2
        s = max(max(hi - lo) * 1.3, 0.35 * min(w, h))
        x0, x1, y0, y1 = cx - s / 2, cx + s / 2, cy - s / 2, cy + s / 2
        x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1))
        if x1 - x0 < 8 or y1 - y0 < 8:
            x0, y0, x1, y1 = 0, 0, w, h
    return cv2.cvtColor(cv2.resize(frame[y0:y1, x0:x1], (size, size)), cv2.COLOR_BGR2RGB)


def frame_active(P, M):
    """True when a hand is visible and raised (not resting at the signer's side)."""
    if not (M[LH].any() or M[RH].any()):
        return False
    if M[L_SH] and M[R_SH]:
        sh_y = (P[L_SH, 1] + P[R_SH, 1]) / 2
        sw = abs(P[L_SH, 0] - P[R_SH, 0]) + 1e-6
        ys = [P[sl, 1].mean() for sl in (LH, RH) if M[sl].any()]
        return min(ys) < sh_y + 1.6 * sw
    return True


# ------------------------------------------------- normalisation / keyframes
def normalize_sequence(P, M):
    """Signer-independence step 1: centre on shoulders, scale by shoulder width."""
    ok = M[:, L_SH] & M[:, R_SH]
    if ok.sum() == 0:
        return None
    sh = P[ok][:, [L_SH, R_SH]]
    centre = np.median(sh.mean(1), axis=0)
    width = max(float(np.median(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1))), 1e-3)
    N = (P - centre) / width
    N[~M] = 0
    return N.astype(np.float32)


def select_keyframes(N, M, k=C.K_FRAMES):
    """Pick k frames spaced evenly along cumulative *motion* (more frames where
    the hands move a lot, fewer during pauses). Blended with uniform spacing."""
    T = len(N)
    if T < 2:
        return np.zeros(k, int)
    valid = M[1:] & M[:-1]
    d = (np.linalg.norm(np.diff(N, axis=0), axis=-1) * valid).sum(-1)
    w = np.ones(T - 1) if d.sum() < 1e-6 else 0.7 * d / d.sum() + 0.3 / (T - 1)
    cum = np.concatenate([[0], np.cumsum(w)])
    idx = np.interp(np.linspace(0, cum[-1], k), cum, np.arange(T))
    return np.round(idx).astype(int)


def build_sample(P, M, crops, encoder=None):
    """P (T,67,2) pixel coords, M (T,67) bool, crops: list of RGB images.
    Returns (skel (K,67,2), rgb (K,576) or None)."""
    act = np.array([frame_active(p, m) for p, m in zip(P, M)])
    if act.any():
        a, b = np.where(act)[0][[0, -1]]
        a, b = max(0, a - 3), min(len(P), b + 4)
        P, M, crops = P[a:b], M[a:b], crops[a:b]
    N = normalize_sequence(P, M)
    if N is None:
        return None
    idx = select_keyframes(N, M)
    rgb = encoder([crops[i] for i in idx]) if encoder is not None else None
    return N[idx], rgb


# ------------------------------------------------------------------ features
_PERM = np.arange(N_PTS)
for _a, _b in [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14),
               (15, 16), (17, 18), (19, 20), (21, 22), (23, 24)]:
    _PERM[_a], _PERM[_b] = _b, _a
_PERM[LH], _PERM[RH] = np.arange(46, 67), np.arange(25, 46)


def mirror(B):
    """Swap left/right (simulates left-handed signers)."""
    B = B[:, _PERM].copy()
    B[..., 0] *= -1
    return B


def build_features(B):
    """(K,67,2) normalised skeleton -> (K, 352): body pos + hand shape + velocity."""
    m = (np.abs(B).sum(-1) > 0)[..., None]
    parts = [B.reshape(len(B), -1)]
    for sl in (LH, RH):
        H = B[:, sl] - B[:, sl][:, :1]
        ext = np.abs(H).max(axis=(1, 2), keepdims=True) + 1e-6
        parts.append((H / ext * m[:, sl]).reshape(len(B), -1))
    vel = np.zeros_like(B)
    vel[1:] = (B[1:] - B[:-1]) * (m[1:] & m[:-1])
    parts.append(vel.reshape(len(B), -1))
    return np.concatenate(parts, 1).astype(np.float32)


def augment_skel(B, rng=np.random):
    """Training-time augmentation for signer variation."""
    m = np.abs(B).sum(-1) > 0
    if rng.rand() < 0.5:
        B, m = mirror(B), m[:, _PERM]
    K = len(B)
    t = np.sort(np.clip(np.arange(K) + rng.normal(0, 0.4, K), 0, K - 1))
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, K - 1)
    a = (t - i0)[:, None, None]
    B = B[i0] * (1 - a) + B[i1] * a
    m = m[i0] & m[i1]
    th = np.deg2rad(rng.uniform(-12, 12))
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    B = (B @ R.T) * rng.uniform(0.85, 1.15, 2) + rng.uniform(-0.15, 0.15, 2)
    B = B + rng.normal(0, 0.01, B.shape)
    return (B * m[..., None]).astype(np.float32)


# ------------------------------------------------------------ RGB embeddings
class RGBEncoder:
    """Frozen MobileNetV3-small -> 576-d embedding per RGB keyframe."""
    def __init__(self):
        import torch
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        self.torch = torch
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.net = torch.nn.Sequential(m.features, torch.nn.AdaptiveAvgPool2d(1),
                                       torch.nn.Flatten()).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __call__(self, crops):
        t = self.torch
        with t.no_grad():
            x = t.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float() / 255
            return self.net((x - self.mean) / self.std).numpy().astype(np.float32)


# ------------------------------------------------- video + live segmentation
def extract_video(path):
    """Read a video -> (P, M, crops) for every frame."""
    cap, ext = cv2.VideoCapture(path), LandmarkExtractor()
    Ps, Ms, crops = [], [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = resize_w(f)
        P, M, _ = ext(f)
        Ps.append(P); Ms.append(M); crops.append(crop_roi(f, P, M))
    cap.release(); ext.close()
    return np.array(Ps), np.array(Ms), crops


class Segmenter:
    """Cuts a continuous webcam stream into individual signs:
    starts when hands are raised, ends after the hands rest for `end` frames."""
    def __init__(self, start=3, end=10, min_len=10, max_len=150):
        self.start, self.end, self.min_len, self.max_len = start, end, min_len, max_len
        self.reset()

    def reset(self):
        self.pre, self.buf, self.run, self.idle, self.rec = deque(maxlen=self.start + 3), [], 0, 0, False

    @property
    def state(self):
        return "Recording sign..." if self.rec else "Waiting (raise your hands to sign)"

    def push(self, P, M, crop):
        item, act = (P, M, crop), frame_active(P, M)
        if not self.rec:
            self.pre.append(item)
            self.run = self.run + 1 if act else 0
            if self.run >= self.start:
                self.rec, self.buf, self.idle = True, list(self.pre), 0
            return None
        self.buf.append(item)
        self.idle = 0 if act else self.idle + 1
        if self.idle >= self.end or len(self.buf) >= self.max_len:
            seg = self.buf[:-self.idle] if self.idle else self.buf
            self.reset()
            return seg if len(seg) >= self.min_len else None
        return None


# ----------------------------------------------------------------- inference
class Predictor:
    def __init__(self, path=C.MODEL_PATH):
        import torch
        from model import SignNet
        self.torch = torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.classes, self.use_rgb = ck["classes"], ck["use_rgb"]
        self.model = SignNet(len(self.classes), hid=ck["hid"], use_rgb=self.use_rgb)
        self.model.load_state_dict(ck["state"]); self.model.eval()
        self.enc = RGBEncoder() if self.use_rgb else None

    def predict_segment(self, seg):
        """seg: list of (P, M, crop). Returns class-probability vector or None."""
        P = np.stack([s[0] for s in seg]); M = np.stack([s[1] for s in seg])
        return self.predict_arrays(P, M, [s[2] for s in seg])

    def predict_arrays(self, P, M, crops):
        out = build_sample(P, M, crops, self.enc)
        if out is None:
            return None
        skel, rgb = out
        t = self.torch
        x = t.from_numpy(build_features(skel))[None]
        r = t.from_numpy(rgb)[None] if rgb is not None else t.zeros(1, C.K_FRAMES, C.RGB_DIM)
        with t.no_grad():
            return t.softmax(self.model(x, r), -1)[0].numpy()

    def predict_video(self, path):
        P, M, crops = extract_video(path)
        return self.predict_arrays(P, M, crops) if len(P) else None
