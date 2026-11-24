"""Train. With >=4 signers: SIGNER-INDEPENDENT split (val/test signers never seen in training).
With 1 signer: falls back to a per-recording split (pipeline check only)."""
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
    items.append(dict(file=f, skel=d["skel"], rgb=d["rgb"] if "rgb" in d.files else None,
                      phrase=str(d["phrase"]), signer=str(d["signer"])))
assert items, "No features found. Run extract_features.py first."
if use_rgb:
    assert all(i["rgb"] is not None for i in items), "RGB missing: re-run extraction or use --modality skeleton"
classes = sorted({i["phrase"] for i in items})
signers = sorted({i["signer"] for i in items})
print(f"{len(items)} samples | {len(classes)} phrases | {len(signers)} signers")
def sample_split(its, frac_val, frac_test, rng):
    """Per-phrase random split of recordings (used when there are too few signers)."""
    tr, va, te = [], [], []
    for c in classes:
        g = [i for i in its if i["phrase"] == c]; rng.shuffle(g)
        nte, nva = max(1, round(len(g) * frac_test)), max(1, round(len(g) * frac_val))
        te += g[:nte]; va += g[nte:nte + nva]; tr += g[nte + nva:]
    return tr, va, te


rng = np.random.RandomState(a.seed)
nt, nv = a.test_signers, a.val_signers
if len(signers) >= nt + nv + 2:                      # proper signer-independent split
    split_type = "signer"
    rng.shuffle(signers)
    S_te, S_va = set(signers[:nt]), set(signers[nt:nt + nv])
    tr_items = [i for i in items if i["signer"] not in S_te | S_va]
    va_items = [i for i in items if i["signer"] in S_va]
    te_items = [i for i in items if i["signer"] in S_te]
    print("train signers:", sorted(set(signers) - S_te - S_va), "| val:", sorted(S_va), "| test:", sorted(S_te))
elif len(signers) >= 2:                              # 2-3 signers: hold out 1 signer for test
    split_type = "signer"
    rng.shuffle(signers)
    te_items = [i for i in items if i["signer"] == signers[0]]
    tr_items, va_items, _ = sample_split([i for i in items if i["signer"] != signers[0]], 0.15, 0.0, rng)
    print(f"WARNING: only {len(signers)} signers -> test signer = {signers[0]}, validation = random recordings.")
else:                                                # single signer
    split_type = "repetition"
    tr_items, va_items, te_items = sample_split(items, 0.2, 0.2, rng)
    print("WARNING: only ONE signer -> splitting by recordings. This is NOT a signer-independent "
          "score; it only checks that the pipeline works. Add more signers for a real test.")
print(f"train {len(tr_items)} | val {len(va_items)} | test {len(te_items)} recordings")


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


mk = lambda its, tr: DataLoader(DS(its, tr), batch_size=32, shuffle=tr, drop_last=False)
dl_tr, dl_val, dl_te = mk(tr_items, True), mk(va_items, False), mk(te_items, False)
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
        print(f"epoch {ep:3d} | train acc {tr:.3f} | val acc {va:.3f}")
    if bad >= a.patience:
        print("early stop"); break

model.load_state_dict(best_state)
te, P, Y = run(dl_te)
print(f"\nBEST val acc {best:.3f} | TEST acc ({split_type} split): {te:.3f}")
os.makedirs("models", exist_ok=True)
torch.save(dict(state=best_state, classes=classes, use_rgb=use_rgb, hid=model.hid), C.MODEL_PATH)
json.dump(dict(test_acc=te, val_acc=best, classes=classes, modality=a.modality,
               split=split_type), open("models/metrics.json", "w"), indent=2)
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
