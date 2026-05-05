"""
train_dinov2_seg.py
────────────────────
Complete training pipeline for DINOv2 binary tree segmentation.

Usage
-----
    python train_dinov2_seg.py

Outputs (all written to ./runs/<timestamp>/)
--------------------------------------------
    model_best.pt          best checkpoint by val IoU
    model_final.pt         weights after last epoch
    results.json           per-epoch metrics for plotting
    viz/                   predicted mask images (test set)
"""

import gc
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import psutil

# ──────────────────────────────────────────────
# 0.  Config  — edit these
# ──────────────────────────────────────────────

CFG = dict(
    root_dir    = "./data",
    cache_dir   = "./data/cache_seg",
    run_dir     = f"./runs/dinov2_seg1",
    mini        = True,
    img_size    = 448,
    batch_size  = 4,
    num_workers = 0,
    epochs      = 60,
    lr          = 1e-4,       
    seed        = 42,
    val_fraction= 0.15,
    # known-bad milliontrees indices that cause OOM in __getitem__
    known_bad   = {("test", 0)},
)


def ram_info(tag=""):
    vm = psutil.virtual_memory()
    print(f"[RAM {tag}] {(vm.total-vm.available)/1e9:.1f}/{vm.total/1e9:.1f} GB ({vm.percent:.0f}%)")


# ──────────────────────────────────────────────
# 1.  Cache build  (runs once, then skipped)
# ──────────────────────────────────────────────

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _process_one(image, y, size):
    img = image.float() / 255.0 if image.dtype == torch.uint8 else image.float()
    img = F.interpolate(img.unsqueeze(0), size=(size, size),
                        mode="bilinear", align_corners=False).squeeze(0)
    img = ((img - _MEAN) / _STD).half()
    binary = y.any(dim=0).float()
    del y
    mask = F.interpolate(binary.unsqueeze(0).unsqueeze(0),
                         size=(size, size), mode="nearest").squeeze().to(torch.int8)
    return img, mask


def build_cache(cfg):
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    index_file = os.path.join(cfg["cache_dir"], "split_index.pt")
    if os.path.exists(index_file):
        print(f"Cache found — loading index.")
        return torch.load(index_file, weights_only=False)

    print("Building cache (first run only)…")
    from milliontrees.datasets.TreePolygons import TreePolygonsDataset
    mw = TreePolygonsDataset(root_dir=cfg["root_dir"], mini=cfg["mini"],
                             split_scheme="random")

    def _process_split(subset, split_name):
        paths, skipped = [], 0
        n = len(subset)
        for i in range(n):
            fpath = os.path.join(cfg["cache_dir"], f"{split_name}_{i:05d}.pt")
            if os.path.exists(fpath):
                paths.append(fpath); continue
            if (split_name, i) in cfg["known_bad"]:
                print(f"  [SKIP] {split_name}[{i}]: known bad")
                skipped += 1; continue
            try:
                metadata, image, targets = subset[i]
                y = targets["y"]
                if y.element_size() * int(np.prod(y.shape)) > 500e6:
                    raise MemoryError(f"mask too large")
                img, mask = _process_one(image, y, cfg["img_size"])
                del image, targets, y
                tmp = fpath + ".tmp"
                torch.save({"img": img, "mask": mask}, tmp)
                os.replace(tmp, fpath)
                paths.append(fpath)
                del img, mask
            except Exception as e:
                skipped += 1
                print(f"  [SKIP] {split_name}[{i}]: {type(e).__name__}: {e}")
            finally:
                gc.collect()
        print(f"  {split_name}: {len(paths)} valid / {skipped} skipped")
        return paths

    test_paths = _process_split(mw.get_subset("test"), "test")
    raw_train  = mw.get_subset("train")
    n          = len(raw_train)
    rng        = np.random.default_rng(cfg["seed"])
    perm       = rng.permutation(n)
    n_val      = max(1, int(cfg["val_fraction"] * n))
    val_paths   = _process_split(Subset(raw_train, perm[:n_val].tolist()),  "val")
    train_paths = _process_split(Subset(raw_train, perm[n_val:].tolist()), "train")

    del mw, raw_train; gc.collect()

    index = {"train": train_paths, "val": val_paths, "test": test_paths}
    torch.save(index, index_file)
    print(f"Cache complete — train:{len(train_paths)} val:{len(val_paths)} test:{len(test_paths)}")
    return index


# ──────────────────────────────────────────────
# 2.  Dataset / DataLoaders
# ──────────────────────────────────────────────

class CachedTreeSegDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        d = torch.load(self.paths[idx], weights_only=True)
        return d["img"], d["mask"].long()   # float16, int64


def make_loaders(index, cfg):
    kw = dict(num_workers=cfg["num_workers"], pin_memory=False, persistent_workers=False)
    train_ds = CachedTreeSegDataset(index["train"])
    val_ds   = CachedTreeSegDataset(index["val"])
    test_ds  = CachedTreeSegDataset(index["test"])
    print(f"Sizes — train:{len(train_ds)} val:{len(val_ds)} test:{len(test_ds)}")
    return (
        DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False, **kw),
        test_ds,
    )


# ──────────────────────────────────────────────
# 3.  Model
# ──────────────────────────────────────────────

class DinoV2Segmentation(nn.Module):
    def __init__(self, backbone, feat_dim=384):
        super().__init__()
        self.backbone = backbone
        """
        self.decoder  = nn.Sequential(
            nn.Conv2d(feat_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 2, kernel_size=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 2, kernel_size=1),
        )
        """
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3),          # was 0.1
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),          # add second dropout
            nn.Conv2d(128, 2, kernel_size=1),
        )
        # kaiming init for faster convergence
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")


    def forward(self, x):
        # x: [B, 3, H, W]  float32
        feats        = self.backbone.forward_features(x)
        patch_tokens = feats["x_norm_patchtokens"]   # [B, N, C]
        B, N, C      = patch_tokens.shape
        H = W        = int(N ** 0.5)
        feat_map     = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        logits       = self.decoder(feat_map)
        logits       = F.interpolate(logits, size=(x.shape[2], x.shape[3]),
                                     mode="bilinear", align_corners=False)
        return logits   # [B, 2, H, W]


def build_model():
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    # freeze backbone — only train the small decoder
    for p in backbone.parameters():
        p.requires_grad = False
    model = DinoV2Segmentation(backbone, feat_dim=384)
    return model


# ──────────────────────────────────────────────
# 4.  Metrics
# ──────────────────────────────────────────────

def compute_metrics(logits, masks):
    """
    logits : [B, 2, H, W]
    masks  : [B, H, W]  int64  values {0,1}
    Returns dict of scalar tensors: loss, precision, recall, iou
    """
    #loss = F.cross_entropy(logits, masks)
    pos_frac = masks.float().mean().clamp(0.01, 0.99)
    weight = torch.tensor([pos_frac, 1.0 - pos_frac], device=logits.device)
    ce_loss = F.cross_entropy(logits, masks, weight=weight)

    probs = F.softmax(logits, dim=1)[:, 1]   # foreground prob [B,H,W]
    flat_p = probs.reshape(-1)
    flat_m = masks.float().reshape(-1)
    dice = 1 - (2 * (flat_p * flat_m).sum() + 1) / (flat_p.sum() + flat_m.sum() + 1)

    loss = ce_loss + dice

    preds = logits.argmax(dim=1)   # [B, H, W]

    tp = ((preds == 1) & (masks == 1)).sum().float()
    fp = ((preds == 1) & (masks == 0)).sum().float()
    fn = ((preds == 0) & (masks == 1)).sum().float()

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    iou       = tp / (tp + fp + fn + 1e-6)

    return {"loss": loss, "precision": precision, "recall": recall, "iou": iou}


# ──────────────────────────────────────────────
# 5.  Train / eval loops
# ──────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    totals = {"loss": 0., "precision": 0., "recall": 0., "iou": 0.}
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, masks in loader:
            # float16 → float32 for the model
            images = images.float().to(device)
            masks  = masks.to(device)

            logits  = model(images)
            metrics = compute_metrics(logits, masks)

            if train:
                optimizer.zero_grad()
                metrics["loss"].backward()
                optimizer.step()

            for k, v in metrics.items():
                totals[k] += v.item()
            n += 1

    return {k: v / n for k, v in totals.items()}


# ──────────────────────────────────────────────
# 6.  Visualisation
# ──────────────────────────────────────────────

def save_visualizations(model, test_ds, device, viz_dir, img_size):
    """Save side-by-side: input | ground truth | prediction for each test sample."""
    os.makedirs(viz_dir, exist_ok=True)
    model.eval()

    # ImageNet denorm for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    with torch.no_grad():
        for i in range(len(test_ds)):
            img_f16, mask_gt = test_ds[i]
            img_f32 = img_f16.float().unsqueeze(0).to(device)  # [1,3,H,W]
            logits  = model(img_f32)
            pred    = logits.argmax(dim=1).squeeze().cpu().numpy()

            # denorm for display
            img_disp = (img_f16.float() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_disp);             axes[0].set_title("Image")
            axes[1].imshow(mask_gt.numpy(), cmap="gray"); axes[1].set_title("Ground Truth")
            axes[2].imshow(pred, cmap="gray");    axes[2].set_title("Prediction")
            for ax in axes: ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"test_{i:03d}.png"), dpi=100)
            plt.close()

    print(f"Visualizations saved to {viz_dir}/")


class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best       = 0.0
        self.counter    = 0

    def step(self, val_iou):
        if val_iou > self.best + self.min_delta:
            self.best    = val_iou
            self.counter = 0
            return False   # don't stop
        self.counter += 1
        print(f"  [EarlyStopping] no improvement for {self.counter}/{self.patience} epochs")
        return self.counter >= self.patience   # stop=True

# ──────────────────────────────────────────────
# 7.  Main
# ──────────────────────────────────────────────



def main():
    torch.manual_seed(CFG["seed"])
    os.makedirs(CFG["run_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    ram_info("start")

    # ── data ──
    index = build_cache(CFG)
    train_loader, val_loader, test_loader, test_ds = make_loaders(index, CFG)

    # ── model ──
    model = build_model().to(device)
    """
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["lr"]
    )
    # reduce LR if val loss plateaus for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )"""

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["lr"], weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"] 
    )

    # ── training loop ──
    history   = {"train": [], "val": []}
    best_iou  = 0.0
    stopper = EarlyStopping(patience=8)
    best_path = os.path.join(CFG["run_dir"], "model_best.pth")

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, optimizer, device, train=True)
        val_m   = run_epoch(model, val_loader,   optimizer, device, train=False)
        #scheduler.step(val_m["iou"])
        scheduler.step()
        if stopper.step(val_m["iou"]):
            print(f"Early stopping at epoch {epoch}")
            break
        history["train"].append(train_m)
        history["val"].append(val_m)

        print(
            f"Epoch {epoch:3d}/{CFG['epochs']}  "
            f"train loss:{train_m['loss']:.4f} iou:{train_m['iou']:.4f}  |  "
            f"val loss:{val_m['loss']:.4f} iou:{val_m['iou']:.4f} "
            f"prec:{val_m['precision']:.4f} rec:{val_m['recall']:.4f}  "
            f"[{time.time()-t0:.1f}s]"
        )

        if val_m["iou"] > best_iou:
            best_iou = val_m["iou"]
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ new best IoU {best_iou:.4f} — saved")

    # ── save final model ──
    torch.save(model.state_dict(), os.path.join(CFG["run_dir"], "model_final.pth"))

    # ── test evaluation ──
    print("\nEvaluating on test set…")
    model.load_state_dict(torch.load(best_path, weights_only=True))  # use best
    test_m = run_epoch(model, test_loader, optimizer, device, train=False)
    print(f"Test — loss:{test_m['loss']:.4f} iou:{test_m['iou']:.4f} "
          f"prec:{test_m['precision']:.4f} rec:{test_m['recall']:.4f}")

    # ── save results ──
    results = {
        "config":  {k: str(v) for k, v in CFG.items()},  # sets aren't JSON-serialisable
        "history": history,
        "test":    test_m,
        "best_val_iou": best_iou,
    }
    results_path = os.path.join(CFG["run_dir"], "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # ── visualise test predictions ──
    save_visualizations(
        model, test_ds, device,
        viz_dir=os.path.join(CFG["run_dir"], "viz"),
        img_size=CFG["img_size"],
    )

    ram_info("end")


if __name__ == "__main__":
    main()