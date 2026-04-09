import os
import json
import copy
import time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def dice_score_from_logits(logits, masks, eps=1e-7):
    probs = torch.softmax(logits, dim=1)[:, 1]
    preds = (probs >= 0.5).float()
    masks = masks.float()
    inter = (preds * masks).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + masks.sum(dim=(1, 2))
    return ((2 * inter + eps) / (union + eps)).mean().item()

def iou_score_from_logits(logits, masks, eps=1e-7):
    probs = torch.softmax(logits, dim=1)[:, 1]
    preds = (probs >= 0.5).float()
    masks = masks.float()
    inter = (preds * masks).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + masks.sum(dim=(1, 2)) - inter
    return ((inter + eps) / (union + eps)).mean().item()


@torch.no_grad()
def evaluate_segmentation(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0
    dice_vals = []
    iou_vals = []

    for images, masks in tqdm(loader):
        images = images.to(device)
        masks = masks.to(device).long()

        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        correct_pixels += (preds == masks).sum().item()
        total_pixels += masks.numel()

        dice_vals.append(dice_score_from_logits(logits, masks))
        iou_vals.append(iou_score_from_logits(logits, masks))

    return {
        "loss": total_loss / max(1, len(loader)),
        "pixel_acc": correct_pixels / max(1, total_pixels),
        "dice": float(np.mean(dice_vals)) if dice_vals else float("nan"),
        "iou": float(np.mean(iou_vals)) if iou_vals else float("nan"),
    }

def train_one_epoch_segmentation(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0
    dice_vals = []
    iou_vals = []

    for images, masks in tqdm(loader):
        images = images.to(device)
        masks = masks.to(device).long()

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        correct_pixels += (preds == masks).sum().item()
        total_pixels += masks.numel()

        dice_vals.append(dice_score_from_logits(logits, masks))
        iou_vals.append(iou_score_from_logits(logits, masks))

    return {
        "loss": total_loss / max(1, len(loader)),
        "pixel_acc": correct_pixels / max(1, total_pixels),
        "dice": float(np.mean(dice_vals)) if dice_vals else float("nan"),
        "iou": float(np.mean(iou_vals)) if iou_vals else float("nan"),
    }


def fit_segmentation(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    run_dir,
    epochs=15,
    scheduler=None,
    patience=3,
):
    ensure_dir(run_dir)

    history = []
    best_val_dice = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch_segmentation(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics = evaluate_segmentation(
            model, val_loader, criterion, device
        )

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_pixel_acc": train_metrics["pixel_acc"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_pixel_acc": val_metrics["pixel_acc"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "time_sec": round(time.time() - t0, 2),
        }
        history.append(row)

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_dice": best_val_dice,
                    "history": history,
                },
                os.path.join(run_dir, "best_model.pth"),
            )
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} dice {train_metrics['dice']:.4f} iou {train_metrics['iou']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} dice {val_metrics['dice']:.4f} iou {val_metrics['iou']:.4f} | "
            f"best dice {best_val_dice:.4f}"
        )

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_segmentation(model, test_loader, criterion, device)

    results = {
        "best_val_dice": best_val_dice,
        "history": history,
        "test": test_metrics,
    }

    save_json(results, os.path.join(run_dir, "results.json"))
    pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_dice": best_val_dice,
            "history": history,
        },
        os.path.join(run_dir, "last_model.pt"),
    )

    return results, model