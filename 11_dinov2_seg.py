import os
import gc
import psutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

IMG_SIZE   = 448          # must be divisible by DINOv2 patch_size (14)
BATCH_SIZE = 4            # safe starting point; lower to 2 if you still OOM
NUM_WORKERS = 0           # 0 = single process, safest in notebooks
SEED       = 42

def ram_info(tag=""):
    vm = psutil.virtual_memory()
    used  = (vm.total - vm.available) / 1e9
    total = vm.total / 1e9
    print(f"[RAM {tag}] {used:.1f} / {total:.1f} GB  ({vm.percent}%)")

image_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class TreeSegDataset(Dataset):
    def __init__(self, base_subset, image_transform=image_transform,
                 mask_size=(IMG_SIZE, IMG_SIZE)):
        self.base      = base_subset
        self.transform = image_transform
        self.mask_size = mask_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # milliontrees returns (metadata, image, targets)
        # image  : PIL Image  (H×W, RGB)
        # targets: dict with key "y" → FloatTensor [N, H, W]  (instance masks)
        metadata, image, targets = self.base[idx]

        # ── image ──────────────────────────────────────────────────────────
        img_tensor = self.transform(image)          # [3, IMG_SIZE, IMG_SIZE]

        # ── binary mask ────────────────────────────────────────────────────
        instance_masks = targets["y"]               # [N, H, W]

        # Collapse instance dim → binary (0/1), keep as float for interpolation
        binary = instance_masks.any(dim=0).float()  # [H, W]

        # Resize mask to match image size (nearest to preserve 0/1)
        binary = F.interpolate(
            binary.unsqueeze(0).unsqueeze(0),       # [1, 1, H, W]
            size=self.mask_size,
            mode="nearest",
        ).squeeze().long()                          # [IMG_SIZE, IMG_SIZE]  int64

        # Free the heavy instance mask immediately
        del instance_masks, targets
        gc.collect()

        return img_tensor, binary

def make_loaders(
    root_dir: str = "./data",
    mini: bool = True,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    val_fraction: float = 0.15,
    seed: int = SEED,
):
    from milliontrees.datasets.TreePolygons import TreePolygonsDataset

    ram_info("before loading milliontrees")

    mw_dataset = TreePolygonsDataset(
        root_dir=root_dir,
        mini=mini,
        split_scheme="random",
    )

    ram_info("after loading milliontrees")

    # ── try official splits, fall back to manual if not present ────────────
    available_splits = mw_dataset.split_dict.keys() \
        if hasattr(mw_dataset, "split_dict") else []

    if "val" in available_splits:
        print("Using official train / val / test splits.")
        raw_train = mw_dataset.get_subset("train")
        raw_val   = mw_dataset.get_subset("val")
        raw_test  = mw_dataset.get_subset("test")

    else:
        print("No official val split found — using manual index split.")
        raw_train_full = mw_dataset.get_subset("train")
        raw_test       = mw_dataset.get_subset("test")

        n_total = len(raw_train_full)
        rng     = np.random.default_rng(seed)
        indices = rng.permutation(n_total)

        n_val     = max(1, int(val_fraction * n_total))
        val_idx   = indices[:n_val].tolist()
        train_idx = indices[n_val:].tolist()

        # Use Subset (index view only — no data copied)
        raw_train = Subset(raw_train_full, train_idx)
        raw_val   = Subset(raw_train_full, val_idx)

    # ── wrap in our lazy binary-mask dataset ───────────────────────────────
    train_ds = TreeSegDataset(raw_train, image_transform=image_transform)
    val_ds   = TreeSegDataset(raw_val,   image_transform=image_transform)
    test_ds  = TreeSegDataset(raw_test,  image_transform=image_transform)

    print(f"Split sizes  →  train: {len(train_ds)}  |  val: {len(val_ds)}  |  test: {len(test_ds)}")

    # ── DataLoaders ────────────────────────────────────────────────────────
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=False,          # avoids extra RAM copy
        persistent_workers=False,  # safe for num_workers=0
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds)

def smoke_test(train_loader, val_loader, test_loader):
    """
    Iterate one batch from each loader and print shapes.
    If this passes without a kernel crash, your pipeline is healthy.
    """
    for name, loader in [("train", train_loader),
                          ("val",   val_loader),
                          ("test",  test_loader)]:
        images, masks = next(iter(loader))
        assert images.ndim == 4,          f"{name}: expected 4-D image tensor"
        assert masks.ndim  == 2 or masks.ndim == 3, \
                                           f"{name}: expected 2-D or 3-D mask"
        assert images.shape[-2:] == torch.Size([IMG_SIZE, IMG_SIZE]), \
            f"{name}: wrong image spatial size {images.shape}"
        print(f"  [{name}]  images: {tuple(images.shape)}  "
              f"masks: {tuple(masks.shape)}  "
              f"mask unique values: {masks.unique().tolist()}")

    ram_info("after smoke test")
    print("✓ Smoke test passed.")


train_loader, val_loader, test_loader, datasets = make_loaders(
    root_dir="./data",
    mini=True,
    batch_size=BATCH_SIZE,
)

smoke_test(train_loader, val_loader, test_loader)