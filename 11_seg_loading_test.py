"""
dataset_pipeline.py  —  memory-optimized for 16 GB RAM + GPU
─────────────────────────────────────────────────────────────
Key changes vs previous version
--------------------------------
* IMG_SIZE = 224 (was 448): 4× less memory for images AND masks.
  DINOv2 still works fine — you get 16×16 = 256 patch tokens per image.
  Change back to 448 only if you have >32 GB RAM.
* Masks collapsed with .any() then immediately deleted — never hold [N,H,W]
  in memory longer than one line.
* Images cast to float16 in the dataset → 2× less RAM for the batch buffer.
  Cast back to float32 inside the training loop before the forward pass.
* BATCH_SIZE = 2 default (was 4). Raise if memory allows.
* transform applied directly in __getitem__ using F.interpolate (no PIL path).
"""

import gc
import psutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

# ──────────────────────────────────────────────
# 0.  Constants  (tune these first if you OOM)
# ──────────────────────────────────────────────

IMG_SIZE    = 224   # 224 → 16 patches/side  |  448 → 32 patches/side (needs ~32 GB)
BATCH_SIZE  = 2     # raise to 4 only if smoke_test passes comfortably
NUM_WORKERS = 0     # MUST be 0 on 16 GB machines — subprocesses fork RAM
SEED        = 42


def ram_info(tag=""):
    vm = psutil.virtual_memory()
    used  = (vm.total - vm.available) / 1e9
    total = vm.total / 1e9
    print(f"[RAM {tag}] {used:.1f} / {total:.1f} GB  ({vm.percent:.0f}%)")


# ──────────────────────────────────────────────
# 1.  Image normalisation constants
# ──────────────────────────────────────────────

# milliontrees returns torch.Tensor [C, H, W]  NOT a PIL Image.
# We resize with F.interpolate and normalise manually — no torchvision needed.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _process_image(img: torch.Tensor, size: int) -> torch.Tensor:
    """
    [C, H, W] uint8 or float32  →  [C, size, size] float16, ImageNet-normed.

    float16 halves the per-batch RAM cost vs float32.
    Cast back to float32 in the training loop: images = images.float()
    """
    img = img.float() / 255.0 if img.dtype == torch.uint8 else img.float()

    img = F.interpolate(
        img.unsqueeze(0),               # [1, C, H, W]
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)                        # [C, size, size]

    img = (img - _MEAN) / _STD
    return img.half()                   # float16 — saves ~50% batch RAM


# ──────────────────────────────────────────────
# 2.  Dataset
# ──────────────────────────────────────────────

class TreeSegDataset(Dataset):
    """
    Returns (image, binary_mask) where:
        image       float16  [3, IMG_SIZE, IMG_SIZE]   ImageNet-normalised
        binary_mask int64    [IMG_SIZE, IMG_SIZE]       values in {0, 1}

    Instance masks [N, H, W] are collapsed to binary immediately and
    freed before the item is returned — never accumulate in the batch buffer.
    """

    def __init__(self, base_subset, img_size: int = IMG_SIZE):
        self.base     = base_subset
        self.img_size = img_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        metadata, image, targets = self.base[idx]

        # ── image ──────────────────────────────────────────────────────────
        img_tensor = _process_image(image, self.img_size)   # float16 [3,S,S]
        del image

        # ── binary mask ────────────────────────────────────────────────────
        # targets["y"] is [N, H, W] — potentially very large.
        # .any(dim=0) collapses N immediately; delete targets right after.
        binary = targets["y"].any(dim=0).float()            # [H, W]
        del targets
        gc.collect()

        binary = F.interpolate(
            binary.unsqueeze(0).unsqueeze(0),               # [1,1,H,W]
            size=(self.img_size, self.img_size),
            mode="nearest",
        ).squeeze().long()                                  # [S, S] int64

        return img_tensor, binary


# ──────────────────────────────────────────────
# 3.  DataLoader factory
# ──────────────────────────────────────────────

def make_loaders(
    root_dir: str       = "./data",
    mini: bool          = True,
    batch_size: int     = BATCH_SIZE,
    num_workers: int    = NUM_WORKERS,
    val_fraction: float = 0.15,
    seed: int           = SEED,
    img_size: int       = IMG_SIZE,
):
    """
    Returns train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds).

    Split logic
    -----------
    1. Uses official 'val' split if milliontrees provides one.
    2. Otherwise carves val out of the train indices using Subset
       (index-only view, zero extra memory).
    """
    from milliontrees.datasets.TreePolygons import TreePolygonsDataset

    ram_info("before loading milliontrees")

    mw_dataset = TreePolygonsDataset(
        root_dir=root_dir,
        mini=mini,
        split_scheme="random",
    )

    ram_info("after loading milliontrees")

    available = set(getattr(mw_dataset, "split_dict", {}).keys())

    if "val" in available:
        print("Using official train / val / test splits.")
        raw_train = mw_dataset.get_subset("train")
        raw_val   = mw_dataset.get_subset("val")
        raw_test  = mw_dataset.get_subset("test")
    else:
        print("No official val split — carving val from train indices.")
        raw_train_full = mw_dataset.get_subset("train")
        raw_test       = mw_dataset.get_subset("test")

        n       = len(raw_train_full)
        rng     = np.random.default_rng(seed)
        idx     = rng.permutation(n)
        n_val   = max(1, int(val_fraction * n))

        raw_val   = Subset(raw_train_full, idx[:n_val].tolist())
        raw_train = Subset(raw_train_full, idx[n_val:].tolist())

    train_ds = TreeSegDataset(raw_train, img_size=img_size)
    val_ds   = TreeSegDataset(raw_val,   img_size=img_size)
    test_ds  = TreeSegDataset(raw_test,  img_size=img_size)

    print(f"Split sizes  →  train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    kw = dict(num_workers=num_workers, pin_memory=False, persistent_workers=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw)

    return train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds)


# ──────────────────────────────────────────────
# 4.  Smoke-test
# ──────────────────────────────────────────────

def smoke_test(train_loader, val_loader, test_loader):
    """One batch per loader — validate shapes, dtypes, mask values."""
    for name, loader in [("train", train_loader),
                          ("val",   val_loader),
                          ("test",  test_loader)]:
        images, masks = next(iter(loader))

        assert images.ndim  == 4,               f"{name}: images {images.shape}"
        assert masks.ndim   == 3,               f"{name}: masks {masks.shape}"
        assert images.dtype == torch.float16,   f"{name}: images dtype {images.dtype}"
        assert masks.dtype  == torch.int64,     f"{name}: masks dtype {masks.dtype}"
        assert set(masks.unique().tolist()).issubset({0, 1}), \
            f"{name}: unexpected mask values {masks.unique().tolist()}"

        print(f"  [{name}]  images: {tuple(images.shape)}  "
              f"masks: {tuple(masks.shape)}  "
              f"mask values: {masks.unique().tolist()}")

    ram_info("after smoke test")
    print("Smoke test passed.")


# ──────────────────────────────────────────────
# 5.  Training-loop usage note
# ──────────────────────────────────────────────
#
# Images come out as float16. In your training loop do:
#
#   for images, masks in train_loader:
#       images = images.float().to(device)   # float16 → float32 for the model
#       masks  = masks.to(device)
#       logits = model(images)               # [B, 2, H, W]
#       loss   = criterion(logits, masks)
#       ...
#
# If you use torch.cuda.amp (mixed precision), you can skip the .float() cast
# and let the autocast handle it — that's even more memory-efficient.


if __name__ == "__main__":
    train_loader, val_loader, test_loader, datasets = make_loaders(
        root_dir="./data",
        mini=True,
        batch_size=BATCH_SIZE,
    )
    smoke_test(train_loader, val_loader, test_loader)