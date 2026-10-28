"""Train with a signer-independent split, or a per-phrase split for one signer."""
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
rng = np.random.RandomState(a.seed)
if len(signers) == 1:
    # With one signer, reserve examples from every phrase instead of requiring
    # other people. This keeps validation/test meaningful without leaking a
    # phrase entirely out of either split.
    by_phrase = {phrase: [i for i in items if i["phrase"] == phrase] for phrase in classes}
    too_few = [phrase for phrase, examples in by_phrase.items() if len(examples) < 3]
    assert not too_few, "Single-signer training needs at least 3 samples per phrase: " + ", ".join(too_few)
    train_items, val_items, test_items = [], [], []
    for examples in by_phrase.values():
        rng.shuffle(examples)
        test_items.append(examples[0])
        val_items.append(examples[1])
        train_items.extend(examples[2:])
    train_s, val_s, test_s = train_items, val_items, test_items
    split_name = "single-signer sample split"
    print("using one-signer mode: held-out samples per phrase for validation and test")
else:
    nt, nv = a.test_signers, a.val_signers
    assert len(signers) >= nt + nv + 1, (
        f"Need at least {nt + nv + 1} signers for the requested signer split; "
        "use one signer with at least 3 samples per phrase."
    )
    rng.shuffle(signers)
    test_s, val_s, train_s = set(signers[:nt]), set(signers[nt:nt + nv]), set(signers[nt + nv:])
    train_items = [i for i in items if i["signer"] in train_s]
    val_items = [i for i in items if i["signer"] in val_s]
    test_items = [i for i in items if i["signer"] in test_s]
    split_name = "signer-independent split"
    print("train signers:", sorted(train_s), "| val:", sorted(val_s), "| test:", sorted(test_s))
print("split:", split_name)


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
dl_tr, dl_val, dl_te = mk(train_items, True), mk(val_items, False), mk(test_items, False)
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
        print(f"epoch {ep:3d} | train acc {tr:.3f} | val acc ({split_name}) {va:.3f}")
    if bad >= a.patience:
        print("early stop"); break

model.load_state_dict(best_state)
te, P, Y = run(dl_te)
print(f"\nBEST val acc {best:.3f} | TEST acc ({split_name}): {te:.3f}")
os.makedirs("models", exist_ok=True)
torch.save(dict(state=best_state, classes=classes, use_rgb=use_rgb, hid=model.hid), C.MODEL_PATH)
json.dump(dict(test_acc=te, val_acc=best, classes=classes, modality=a.modality,
               split=split_name,
               test_signers=sorted(test_s) if isinstance(test_s, set) else []),
          open("models/metrics.json", "w"), indent=2)
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
