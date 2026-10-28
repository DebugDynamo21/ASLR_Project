# Signer-Independent Sign Phrase Recognition (keyframes + skeleton)

**Pipeline:** video → MediaPipe Holistic (pose + both hands) → trim idle frames → **motion-adaptive keyframes (24)**
→ skeleton features (shoulder-normalised, hand-shape, velocity) + RGB keyframe crops (MobileNetV3 embeddings)
→ two-stream BiGRU, late fusion → phrase.

**Why it's signer-independent:** shoulder-width normalisation, hand-local shape features, left/right mirror + scale/rotation/shift/time
augmentation, RGB only on a body crop with modality-dropout, and — most important — the train/val/test split is **by signer**.

## 1. Setup (Python 3.10 or 3.11)
```
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get data  (this is what decides your accuracy)
Edit `PHRASES` in `config.py` (15–20 phrases), then have **at least 8–10 different people** record each phrase **10+ times**:
```
python collect_data.py --signer alice --reps 10
python collect_data.py --signer bob   --reps 10
```
Videos land in `data/raw/<phrase>/<signer>_<n>.mp4`. You can also drop in videos from an existing dataset
(e.g. WLASL / ISL / your own) using the same naming: `data/raw/<phrase>/<signerid>_<n>.mp4`.
Tips: vary background, lighting, distance, clothing; include a left-handed signer; show upper body + hands fully; lower hands between signs.

## 3. Extract keyframe features
```
python extract_features.py          # add --no-rgb for skeleton only
```
## 4. Train (signer-independent split)
```
python train.py                      # skeleton + RGB
python train.py --modality skeleton  # compare: often equally good and more robust
```
Prints accuracy on **unseen signers**, saves `models/sign_model.pt`, `models/metrics.json`, `models/confusion_matrix.png`.
Use whichever modality gives the better *unseen-signer* accuracy.

## 5. Run the UI
```
streamlit run app.py
```
Live webcam mode segments signs automatically (hands up = start, hands down = result). Unknown/low-confidence → "unknown" (tune slider).

## 6. Deploy on Streamlit Community Cloud
Push this folder (including `models/sign_model.pt`, `requirements.txt`, `packages.txt`) to GitHub → share.streamlit.io →
Advanced settings → choose **Python 3.11**. Webcam over WebRTC may need a TURN server on some networks.

## Growing the vocabulary later
Add folders/phrases, record more signers, re-run steps 3–4. Consider adding a "no-sign / other" class of random movement for fewer false positives.
