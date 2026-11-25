"""Train the sign recognizer with either signer or sample based evaluation splits."""

# Standard-library modules handle command-line options, file discovery, JSON output,
# and creation of the directory that stores the trained model.
import argparse, glob, json, os, tempfile

# NumPy prepares feature arrays and random operations; PyTorch trains the network.
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Project modules provide configuration values, the neural network, and feature logic.
import config as C
from model import SignNet
from signlib import augment_skel, build_features

# Define the command-line interface so training settings can be changed without
# editing this file.
ap = argparse.ArgumentParser()
ap.add_argument("--modality", choices=["both", "skeleton"], default="both")  # Select skeleton only or skeleton plus RGB features.
ap.add_argument("--epochs", type=int, default=120)  # Maximum number of complete passes through the training data.
ap.add_argument("--patience", type=int, default=25)  # Stop after this many non-improving validation epochs.
ap.add_argument("--test-signers", type=int, default=2)  # Number of unseen signers reserved for final testing.
ap.add_argument("--val-signers", type=int, default=2)  # Number of unseen signers reserved for validation.
ap.add_argument("--seed", type=int, default=0)  # Seed used to make data splitting and training repeatable.
a = ap.parse_args()

# Seed both random libraries so repeated runs use the same split and initialization.
np.random.seed(a.seed); torch.manual_seed(a.seed)

# RGB is enabled only for the combined modality; otherwise zero RGB inputs are used.
use_rgb = a.modality == "both"

# Prefer the GPU when PyTorch can access one, otherwise train on the CPU.
dev = "cuda" if torch.cuda.is_available() else "cpu"

# Load every extracted feature file from the processed-data directory.
items = []
for f in glob.glob(f"{C.PROC_DIR}/*.npz"):
    # Read the compressed NumPy archive produced by extract_features.py.
    d = np.load(f)
    # Keep the required skeleton and metadata, and keep RGB only when it exists.
    items.append(dict(skel=d["skel"], rgb=d["rgb"] if "rgb" in d.files else None,
                      phrase=str(d["phrase"]), signer=str(d["signer"])))

# Training cannot start without extracted samples.
assert items, "No features found. Run extract_features.py first."
if use_rgb:
    # Combined training requires RGB embeddings for every sample to keep batches uniform.
    assert all(i["rgb"] is not None for i in items), "RGB missing: re-run extraction or use --modality skeleton"

# Build stable class and signer lists used for labels and split construction.
classes = sorted({i["phrase"] for i in items})
signers = sorted({i["signer"] for i in items})
print(f"{len(items)} samples | {len(classes)} phrases | {len(signers)} signers")

# Use an isolated random generator for shuffling split candidates.
rng = np.random.RandomState(a.seed)
if len(signers) == 1:
    # With one signer, reserve examples from every phrase instead of requiring
    # other people. This keeps validation/test meaningful without leaking a
    # phrase entirely out of either split.
    # Group samples by phrase so every phrase contributes to all three splits.
    by_phrase = {phrase: [i for i in items if i["phrase"] == phrase] for phrase in classes}
    # One sample is needed for validation, one for testing, and at least one for training.
    too_few = [phrase for phrase, examples in by_phrase.items() if len(examples) < 3]
    assert not too_few, "Single-signer training needs at least 3 samples per phrase: " + ", ".join(too_few)
    train_items, val_items, test_items = [], [], []
    for examples in by_phrase.values():
        # Shuffle each phrase independently before selecting held-out examples.
        rng.shuffle(examples)
        # Hold out distinct examples for validation and final testing.
        test_items.append(examples[0])
        val_items.append(examples[1])
        # Train on all remaining examples for this phrase.
        train_items.extend(examples[2:])
    # In single-signer mode the split labels are sample lists, not signer sets.
    train_s, val_s, test_s = train_items, val_items, test_items
    split_name = "single-signer sample split"
    print("using one-signer mode: held-out samples per phrase for validation and test")
else:
    # In the normal mode, reserve complete signers so evaluation measures generalization
    # to people whose examples were never seen during training.
    nt, nv = a.test_signers, a.val_signers
    assert len(signers) >= nt + nv + 1, (
        f"Need at least {nt + nv + 1} signers for the requested signer split; "
        "use one signer with at least 3 samples per phrase."
    )
    # Shuffle signer IDs before assigning non-overlapping train, validation, and test groups.
    rng.shuffle(signers)
    test_s, val_s, train_s = set(signers[:nt]), set(signers[nt:nt + nv]), set(signers[nt + nv:])
    # Select every extracted sample belonging to each signer's assigned group.
    train_items = [i for i in items if i["signer"] in train_s]
    val_items = [i for i in items if i["signer"] in val_s]
    test_items = [i for i in items if i["signer"] in test_s]
    split_name = "signer-independent split"
    print("train signers:", sorted(train_s), "| val:", sorted(val_s), "| test:", sorted(test_s))
print("split:", split_name)


class DS(Dataset):
    """Convert extracted dictionaries into tensors consumed by the model."""

    def __init__(self, its, train):
        # Store the selected split and whether training-only augmentation is enabled.
        self.its, self.train = its, train

    def __len__(self):
        # DataLoader uses this count to determine the number of available samples.
        return len(self.its)

    def __getitem__(self, k):
        # Retrieve one precomputed sample by its dataset index.
        d = self.its[k]
        # Randomly perturb skeletons only during training to improve robustness.
        B = augment_skel(d["skel"]) if self.train else d["skel"]
        # Occasionally replace RGB with zeros during training (modality dropout), while
        # keeping real RGB features for evaluation whenever RGB mode is enabled.
        if use_rgb and not (self.train and np.random.rand() < 0.3):
            # Add small training noise to RGB embeddings so the model does not memorize
            # exact feature values from the training set.
            r = d["rgb"] + (np.random.normal(0, 0.05, d["rgb"].shape) if self.train else 0)
        else:
            # Match the expected RGB tensor shape when RGB is disabled or dropped out.
            r = np.zeros((C.K_FRAMES, C.RGB_DIM))
        # Convert features to tensors and map the phrase string to its class index.
        return (torch.from_numpy(build_features(B)), torch.from_numpy(r).float(),
                classes.index(d["phrase"]))


# Create loaders for each split; only training data is shuffled and augmented.
mk = lambda its, tr: DataLoader(DS(its, tr), batch_size=32, shuffle=tr, drop_last=False)
dl_tr, dl_val, dl_te = mk(train_items, True), mk(val_items, False), mk(test_items, False)

# Construct the two-stream classifier and move its parameters to the chosen device.
model = SignNet(len(classes), use_rgb=use_rgb).to(dev)

# AdamW updates model weights and applies decoupled weight decay for regularization.
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)

# Gradually reduce the learning rate across the requested number of epochs.
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

# Cross-entropy measures phrase-classification error; label smoothing reduces overconfidence.
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)


def run(dl, train=False):
    """Run one data loader pass and return accuracy plus predictions and labels."""

    # Toggle dropout/batch behavior and prepare counters for this pass.
    model.train(train); correct = n = 0; P, Y = [], []
    for x, r, y in dl:
        # Move skeleton features, RGB features, and target labels to the same device.
        x, r, y = x.to(dev), r.to(dev), y.to(dev)
        # Gradients are needed only for training; disabling them saves memory for evaluation.
        with torch.set_grad_enabled(train):
            out = model(x, r); loss = loss_fn(out, y)
        if train:
            # Clear old gradients, backpropagate the current loss, clip unstable gradients,
            # and update the model parameters once for this batch.
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        # The largest output logit is the predicted phrase class.
        p = out.argmax(1); correct += (p == y).sum().item(); n += len(y)
        # Keep predictions and labels for accuracy and the final confusion matrix.
        P += p.tolist(); Y += y.tolist()
    # Avoid division by zero if an empty split is ever supplied.
    return correct / max(n, 1), P, Y


# Track the best validation checkpoint and consecutive epochs without improvement.
best, best_state, bad = -1, None, 0
for ep in range(a.epochs):
    # Train for one epoch, then advance the learning-rate schedule.
    tr, _, _ = run(dl_tr, True); sched.step()
    # Evaluate without updating weights so validation reflects generalization.
    va, _, _ = run(dl_val)
    if va > best:
        # Copy the best weights to CPU so they remain available after later updates.
        best, bad = va, 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        # Count validation epochs that did not improve the best result.
        bad += 1
    if ep % 5 == 0:
        # Print periodic progress without making the training output excessively noisy.
        print(f"epoch {ep:3d} | train acc {tr:.3f} | val acc ({split_name}) {va:.3f}")
    if bad >= a.patience:
        # Stop early when additional epochs are no longer improving validation accuracy.
        print("early stop"); break

# Restore the best validation checkpoint before measuring final test performance.
model.load_state_dict(best_state)
te, P, Y = run(dl_te)
print(f"\nBEST val acc {best:.3f} | TEST acc ({split_name}): {te:.3f}")

# Create the output directory and save weights plus metadata needed by the app.
os.makedirs("models", exist_ok=True)
torch.save(dict(state=best_state, classes=classes, use_rgb=use_rgb, hid=model.hid), C.MODEL_PATH)
metrics = dict(test_acc=te, val_acc=best, classes=classes, modality=a.modality,
               split=split_name,
               test_signers=sorted(test_s) if isinstance(test_s, set) else [])
metrics_path = os.path.join("models", "metrics.json")
fd, metrics_tmp = tempfile.mkstemp(dir="models", prefix=".metrics-", suffix=".tmp")
try:
    with os.fdopen(fd, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.flush()
        os.fsync(metrics_file.fileno())
    os.replace(metrics_tmp, metrics_path)
finally:
    if os.path.exists(metrics_tmp):
        os.unlink(metrics_tmp)

# Optionally produce a human-readable report and confusion matrix when scikit-learn
# and matplotlib are installed. Training results remain available if reporting fails.
try:
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # Use one label per known phrase so missing predictions still appear in the report.
    labs = list(range(len(classes)))
    print(classification_report(Y, P, labels=labs, target_names=classes, zero_division=0))
    # Render and save a matrix showing which phrases are confused with one another.
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(Y, P, labels=labs, display_labels=classes, ax=ax,
                                            xticks_rotation=90, colorbar=False)
    fig.tight_layout(); fig.savefig("models/confusion_matrix.png", dpi=120)
except Exception as e:
    # Report generation is supplementary; do not discard the trained model if it fails.
    print("report skipped:", e)
print("saved", C.MODEL_PATH)
