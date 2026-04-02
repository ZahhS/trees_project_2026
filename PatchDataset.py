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
import pandas as pd

AUG_ROOT = "patches_dataset/patches_v3_train_aug"
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

        y = torch.tensor(label, dtype=torch.long)
        return img, y

def load_dataset(traindf,testdf,PATCHES_ROOT,BATCH_SIZE):
    transform = transforms.Compose([transforms.ToTensor()])

    subtrain_df, val_df = train_test_split(traindf,test_size=0.2,random_state=SEED,stratify=traindf["label"])

    train_dataset = PatchDataset(subtrain_df, root_dir=PATCHES_ROOT, transform=transform)
    val_dataset  = PatchDataset(val_df,  root_dir=PATCHES_ROOT, transform=transform)
    print("train/val:", len(train_dataset), len(val_dataset))

    # loading seperate test dataset
    test_dataset = PatchDataset(testdf, root_dir=PATCHES_ROOT, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader  = DataLoader(val_dataset,  batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_dataset,val_dataset,train_loader,val_loader, test_dataset, test_loader

class AugmentedPatchDataset(Dataset):
    def __init__(self, df, orig_root, aug_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.orig_root = orig_root
        self.aug_root = aug_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patch_id = str(row["patch_id"])
        label = int(row["label"])
        
        subdir = "positive" if label == 1 else "negative"
        
        # Check if its augmented by looking at the suffix of the column type sufix _aug
        if "_aug" in row["type"]:
            img_path = os.path.join(self.aug_root, subdir, f"{patch_id}.png")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing augmented patch file: {img_path}")
        else:
            img_path = os.path.join(self.orig_root, subdir, f"{patch_id}.png")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing: orig={img_path}")
    
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        y = torch.tensor(label, dtype=torch.long)
        return img, y

def load_augmented_dataset(orig_root, aug_root, path_aug_metadata, path_test_metadata, BATCH_SIZE):
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load augmented train (combined original + aug)
    traindf = pd.read_csv(path_aug_metadata)
    subtrain_df, val_df = train_test_split(traindf,test_size=0.2,random_state=SEED,stratify=traindf["label"])

    train_dataset = AugmentedPatchDataset(subtrain_df, orig_root, aug_root, transform)
    val_dataset  = AugmentedPatchDataset(val_df, orig_root, aug_root, transform)

    # Load test (original only)
    test_df = pd.read_csv(path_test_metadata)
    test_dataset = PatchDataset(test_df, orig_root, transform)
    
    print("train_aug/test:", len(train_dataset), len(test_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader  = DataLoader(val_dataset,  batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    return train_dataset,val_dataset,train_loader,val_loader, test_dataset, test_loader