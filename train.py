"""Train with a SIGNER-INDEPENDENT split: validation and test signers never appear in training."""
import argparse, glob, json, os
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import config as C
from model import SignNet
from signlib import augment_skel, build_features

ap = argparse.ArgumentParser()
ap.add_argument("--modality", choices=["both", "skeleton"], default="both")
ap.add_argument("--epochs", type=int, default=120)
ap.add_argument("--patience", type=int, default=25)
ap.add_argument("--test-signers", type=int, default=2)
ap.add_argument("--val-signers", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
np.random.seed(a.seed); torch.manual_seed(a.seed)
use_rgb = a.modality == "both"
dev = "cuda" if torch.cuda.is_available() else "cpu"

items = []
for f in glob.glob(f"{C.PROC_DIR}/*.npz"):
    d = np.load(f)
    items.append(dict(skel=d["skel"], rgb=d["rgb"] if "rgb" in d.files else None,
                      phrase=str(d["phrase"]), signer=str(d["signer"])))
assert items, "No features found. Run extract_features.py first."
if use_rgb:
    assert all(i["rgb"] is not None for i in items), "RGB missing: re-run extraction or use --modality skeleton"
classes = sorted({i["phrase"] for i in items})
signers = sorted({i["signer"] for i in items})
print(f"{len(items)} samples | {len(classes)} phrases | {len(signers)} signers")
nt, nv = a.test_signers, a.val_signers
if len(signers) < nt + nv + 2:
    nt, nv = 1, 1
    assert len(signers) >= 4, "Need at least 4 different signers for a signer-independent split."
rng = np.random.RandomState(a.seed); rng.shuffle(signers)
test_s, val_s, train_s = set(signers[:nt]), set(signers[nt:nt + nv]), set(signers[nt + nv:])
print("train signers:", sorted(train_s), "| val:", sorted(val_s), "| test:", sorted(test_s))
pick = lambda S: [i for i in items if i["signer"] in S]


class DS(Dataset):
    def __init__(self, its, train):
        self.its, self.train = its, train
    def __len__(self):
        return len(self.its)
    def __getitem__(self, k):
        d = self.its[k]
        B = augment_skel(d["skel"]) if self.train else d["skel"]
        if use_rgb and not (self.train and np.random.rand() < 0.3):      # modality dropout
            r = d["rgb"] + (np.random.normal(0, 0.05, d["rgb"].shape) if self.train else 0)
        else:
            r = np.zeros((C.K_FRAMES, C.RGB_DIM))
        return (torch.from_numpy(build_features(B)), torch.from_numpy(r).float(),
                classes.index(d["phrase"]))


mk = lambda S, tr: DataLoader(DS(pick(S), tr), batch_size=32, shuffle=tr, drop_last=False)
dl_tr, dl_val, dl_te = mk(train_s, True), mk(val_s, False), mk(test_s, False)
model = SignNet(len(classes), use_rgb=use_rgb).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)


def run(dl, train=False):
    model.train(train); correct = n = 0; P, Y = [], []
    for x, r, y in dl:
        x, r, y = x.to(dev), r.to(dev), y.to(dev)
        with torch.set_grad_enabled(train):
            out = model(x, r); loss = loss_fn(out, y)
        if train:
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        p = out.argmax(1); correct += (p == y).sum().item(); n += len(y)
        P += p.tolist(); Y += y.tolist()
    return correct / max(n, 1), P, Y


best, best_state, bad = -1, None, 0
for ep in range(a.epochs):
    tr, _, _ = run(dl_tr, True); sched.step()
    va, _, _ = run(dl_val)
    if va > best:
        best, bad = va, 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
    if ep % 5 == 0:
        print(f"epoch {ep:3d} | train acc {tr:.3f} | val acc (unseen signers) {va:.3f}")
    if bad >= a.patience:
        print("early stop"); break

model.load_state_dict(best_state)
te, P, Y = run(dl_te)
print(f"\nBEST val acc {best:.3f} | TEST acc on unseen signers: {te:.3f}")
os.makedirs("models", exist_ok=True)
torch.save(dict(state=best_state, classes=classes, use_rgb=use_rgb, hid=model.hid), C.MODEL_PATH)
json.dump(dict(test_acc=te, val_acc=best, classes=classes, modality=a.modality,
               test_signers=sorted(test_s)), open("models/metrics.json", "w"), indent=2)
try:
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    labs = list(range(len(classes)))
    print(classification_report(Y, P, labels=labs, target_names=classes, zero_division=0))
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(Y, P, labels=labs, display_labels=classes, ax=ax,
                                            xticks_rotation=90, colorbar=False)
    fig.tight_layout(); fig.savefig("models/confusion_matrix.png", dpi=120)
except Exception as e:
    print("report skipped:", e)
print("saved", C.MODEL_PATH)
