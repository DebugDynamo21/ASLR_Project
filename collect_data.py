"""Record training videos with your webcam.
Usage:  python collect_data.py --signer alice --reps 10
Each person ("signer") runs this once with their own id. SPACE = start/stop, S = skip phrase, Q = quit."""
import argparse, os, time
import cv2
import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--signer", required=True, help="unique id, no underscores (e.g. alice)")
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--cam", type=int, default=0)
ap.add_argument("--phrases", nargs="*", default=C.PHRASES)
a = ap.parse_args()
assert "_" not in a.signer, "signer id must not contain '_'"

cap = cv2.VideoCapture(a.cam)
W, H = int(cap.get(3)), int(cap.get(4))
quit_all = False
for phrase in a.phrases:
    d = os.path.join(C.RAW_DIR, phrase); os.makedirs(d, exist_ok=True)
    n = len([f for f in os.listdir(d) if f.startswith(a.signer + "_")])
    while n < a.reps and not quit_all:
        writer, t0 = None, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("Cannot read camera")
            if writer is not None:
                writer.write(frame)                       # raw (un-mirrored) frame
            view = cv2.flip(frame, 1)
            msg = "REC - press SPACE to stop" if writer else "SPACE = start recording"
            cv2.rectangle(view, (0, 0), (W, 70), (0, 0, 0), -1)
            cv2.putText(view, f"{phrase.replace('_',' ').upper()}  ({n+1}/{a.reps})", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(view, msg, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255) if writer else (0, 255, 0), 2)
            cv2.imshow("collect", view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(" "):
                if writer is None:
                    writer = cv2.VideoWriter(os.path.join(d, f"{a.signer}_{n:02d}.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
                    t0 = time.time()
                else:
                    writer.release(); n += 1; break
            elif k == ord("q"):
                if writer: writer.release()
                quit_all = True; break
            elif k == ord("s") and writer is None:
                n = a.reps; break
            if writer is not None and time.time() - t0 > 6:   # safety stop
                writer.release(); n += 1; break
    if quit_all:
        break
cap.release(); cv2.destroyAllWindows()
