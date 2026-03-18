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

    train_df, test_df = train_test_split(df,test_size=0.2,random_state=SEED)

    train_dataset = PatchDataset(train_df, root_dir=PATCHES_ROOT, transform=transform)
    test_dataset  = PatchDataset(test_df,  root_dir=PATCHES_ROOT, transform=transform)

    print("train/test:", len(train_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_dataset,test_dataset,train_loader,test_loader

