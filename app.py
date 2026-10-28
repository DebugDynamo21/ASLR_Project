"""Streamlit UI:  streamlit run app.py"""
import os, json, time, threading, tempfile
from collections import deque
from functools import lru_cache
import av, cv2, numpy as np, streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import config as C
from signlib import LandmarkExtractor, Segmenter, Predictor, resize_w, crop_roi

st.set_page_config(page_title="Sign Language Recognition", page_icon="🤟", layout="wide")


@lru_cache(maxsize=1)
def get_predictor():
    return Predictor(C.MODEL_PATH)


class SignProcessor(VideoProcessorBase):
    def __init__(self):
        self.ext, self.pred, self.seg = LandmarkExtractor(), get_predictor(), Segmenter()
        self.lock, self.threshold = threading.Lock(), 0.6
        self.last, self.history = None, deque(maxlen=10)

    def snapshot(self):
        with self.lock:
            return dict(state=self.seg.state, last=self.last, history=list(self.history))

    def _update(self, p):
        order = np.argsort(p)[::-1]
        top, conf = int(order[0]), float(p[order[0]])
        ok = conf >= self.threshold
        with self.lock:
            self.last = dict(label=self.pred.classes[top] if ok else "unknown", conf=conf, ok=ok,
                             top3=[(self.pred.classes[i], float(p[i])) for i in order[:3]])
            if ok:
                self.history.append(self.pred.classes[top])

    def recv(self, frame):
        img = resize_w(frame.to_ndarray(format="bgr24"))
        P, M, res = self.ext(img)
        segment = self.seg.push(P, M, crop_roi(img, P, M))
        if segment is not None:
            p = self.pred.predict_segment(segment)
            if p is not None:
                self._update(p)
        vis = img.copy()
        mp = self.ext.mp
        for lm, conn in ((res.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS),
                         (res.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS),
                         (res.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)):
            if lm is not None:
                mp.solutions.drawing_utils.draw_landmarks(vis, lm, conn)
        vis = cv2.flip(vis, 1)
        with self.lock:
            txt = self.last["label"].replace("_", " ").upper() if self.last else "..."
            rec = self.seg.rec
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(vis, ("REC  " if rec else "") + txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255) if rec else (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(vis, format="bgr24")


st.title("🤟 Real-Time Sign Phrase Recognition")
st.caption("Skeletal keyframes (MediaPipe) + RGB keyframes (MobileNetV3) → two-stream BiGRU")

if not os.path.exists(C.MODEL_PATH):
    st.error("No trained model found at `models/sign_model.pt`. Run the data → extract → train steps in the README first.")
    st.stop()

pred = get_predictor()
with st.sidebar:
    st.header("Settings")
    thr = st.slider("Confidence threshold", 0.3, 0.95, 0.6, 0.05,
                    help="Below this, the output is shown as 'unknown'.")
    mode = st.radio("Mode", ["Live webcam", "Upload video"])
    st.markdown(f"**Modalities:** {'skeleton + RGB' if pred.use_rgb else 'skeleton only'}")
    if os.path.exists("models/metrics.json"):
        m = json.load(open("models/metrics.json"))
        metric_label = "Accuracy on held-out samples" if m.get("split") == "single-signer sample split" \
            else "Accuracy on unseen signers"
        st.metric(metric_label, f"{m['test_acc']*100:.1f}%")
    with st.expander(f"Supported phrases ({len(pred.classes)})"):
        st.write(", ".join(c.replace("_", " ") for c in pred.classes))

if mode == "Live webcam":
    st.info("Click START, allow the camera, make sure your upper body is visible. "
            "Raise your hands, sign a phrase, then lower your hands — the result appears after you stop.")
    c1, c2 = st.columns([3, 2])
    with c1:
        ctx = webrtc_streamer(
            key="sign", mode=WebRtcMode.SENDRECV, video_processor_factory=SignProcessor,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True)
    with c2:
        status, big, bar, top3, hist = st.empty(), st.empty(), st.empty(), st.empty(), st.empty()
    if ctx.video_processor:
        while ctx.state.playing:
            vp = ctx.video_processor
            vp.threshold = thr
            s = vp.snapshot()
            status.markdown(f"**Status:** {s['state']}")
            if s["last"]:
                l = s["last"]
                big.markdown(f"## {'✅' if l['ok'] else '❓'} {l['label'].replace('_', ' ').title()}")
                bar.progress(min(l["conf"], 1.0), text=f"confidence {l['conf']*100:.0f}%")
                top3.table({"phrase": [t[0].replace("_", " ") for t in l["top3"]],
                            "prob": [f"{t[1]*100:.0f}%" for t in l["top3"]]})
            hist.markdown("**Recognised:** " + (" → ".join(h.replace("_", " ") for h in s["history"]) or "—"))
            time.sleep(0.25)
else:
    up = st.file_uploader("Upload a short video of one signed phrase", type=["mp4", "mov", "avi", "webm"])
    if up:
        st.video(up)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as t:
            t.write(up.read()); path = t.name
        with st.spinner("Analysing..."):
            p = pred.predict_video(path)
        os.unlink(path)
        if p is None:
            st.warning("Could not find a person (shoulders + hands) in the video.")
        else:
            order = np.argsort(p)[::-1][:5]
            lab = pred.classes[order[0]].replace("_", " ").title()
            st.subheader(f"{'✅' if p[order[0]] >= thr else '❓'} {lab}  ({p[order[0]]*100:.0f}%)")
            st.bar_chart({pred.classes[i].replace("_", " "): float(p[i]) for i in order})
