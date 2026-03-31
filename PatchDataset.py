from torch.utils.data import DataLoader,Subset,Dataset
import torch
import torch.nn as nn
import os
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import random
from torchvision import datasets, transforms
from pathlib import Path
import cv2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class PatchDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patch_id = int(row["patch_id"])
        label = int(row["label"])
        t = row["type"].strip().lower()

        subdir = "positive" if t == "positive" else "negative"
        img_path = os.path.join(self.root_dir, subdir, f"{patch_id}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing patch file: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        y = torch.tensor(label, dtype=torch.float32)
        return img, y

def load_dataset(df,PATCHES_ROOT,BATCH_SIZE):
    transform = transforms.Compose([transforms.ToTensor()])

    train_df, test_df = train_test_split(df,test_size=0.2,random_state=SEED,stratify=df["label"])

    train_dataset = PatchDataset(train_df, root_dir=PATCHES_ROOT, transform=transform)
    test_dataset  = PatchDataset(test_df,  root_dir=PATCHES_ROOT, transform=transform)

    print("train/test:", len(train_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_dataset,test_dataset,train_loader,test_loader



class UnifiedPatchDataset(Dataset):
    def __init__(self, df, orig_root, aug_root=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.orig_root = Path(orig_root)
        self.aug_root = Path(aug_root) if aug_root else None
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid, label = row["patch_id"], int(row["label"])
        
        # Try augmented path first, then original
        if self.aug_root and str(pid).startswith(tuple(f"{p}_aug_" for p in range(10000))):
            subdir = "positive" if label == 1 else "negative"
            img_path = self.aug_root / subdir / f"{pid}.png"
        else:
            subdir = "positive" if label == 1 else "negative"
            img_path = self.orig_root / subdir / f"{pid}.png"
        
        if not img_path.exists():
            raise FileNotFoundError(f"Missing: {img_path}")
        
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(image=img)["image"]
        
        return img, torch.tensor(label, dtype=torch.float32)



def load_dataset(df,PATCHES_ROOT,BATCH_SIZE):
    transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = UnifiedPatchDataset(test_df, PATCHES_ROOT)
    train_dataset_aug = UnifiedPatchDataset(train_df_combined, PATCHES_ROOT, AUG_ROOT, eval_transform)

    train_df, test_df = train_test_split(df,test_size=0.2,random_state=SEED,stratify=df["label"])

    train_dataset = PatchDataset(train_df, root_dir=PATCHES_ROOT, transform=transform)
    test_dataset  = PatchDataset(test_df,  root_dir=PATCHES_ROOT, transform=transform)

    print("train/test:", len(train_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_dataset,test_dataset,train_loader,test_loader



print("Dataset verification:")
for images, labels in DataLoader(train_dataset_aug, batch_size=4):
    print(f"Batch shape: {images.shape}, labels: {labels}")
    break