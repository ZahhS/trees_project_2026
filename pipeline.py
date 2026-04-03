import os
import json
import time
import copy
from dataclasses import asdict, dataclass
from xml.parsers.expat import model
import numpy as np
import torch
import random
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score)




def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def logits_to_probs(logits: torch.Tensor):
    """Convert class logits to positive-class probabilities."""
    return torch.softmax(logits, dim=1)[:, 1]

def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    """Compute binary classification metrics from labels and positive-class probabilities."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "ap": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y_true)),
    }
    return metrics

@torch.no_grad()
def predict_probs(model, loader, device):
    """Collect labels and positive-class probabilities for a dataloader."""
    model.eval()
    all_probs = []
    all_labels = []

    for images, labels in tqdm(loader):
        images = images.to(device)
        outputs = model(images)
        probs = logits_to_probs(outputs).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_probs)

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch and return average loss and batch-aggregated predictions."""
    model.train()
    losses = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        probs = logits_to_probs(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    metrics = compute_binary_metrics(y_true, y_prob)
    metrics["loss"] = float(np.mean(losses))
    return metrics

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on a dataloader and return loss plus metrics."""
    model.eval()
    losses = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        losses.append(loss.item())
        probs = logits_to_probs(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    metrics = compute_binary_metrics(y_true, y_prob)
    metrics["loss"] = float(np.mean(losses))
    return metrics

def save_json(obj, path):
    """Save a JSON-serializable object."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_checkpoint(path, model, optimizer, epoch, config, best_metric=None, history=None):
    """Save a full training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
        "best_metric": best_metric,
        "history": history,
    }
    torch.save(checkpoint, path)

def fit_binary_classifier(model,train_loader,val_loader,test_loader,criterion,optimizer,device,run_dir,config,scheduler=None,patience=5,monitor="f1",
                          min_delta=0.0,threshold=0.5,verbose=True):
    """
    Train a binary classifier, early‑stop on val, save best model, and eval once on test.
    Saves everything into one JSON.
    """
    ensure_dir(run_dir)
    SAVING_FILE_NAME = os.path.join(run_dir, f"best_model_{config.get('model')}_{config.get('learning_rate')}_{config.get('epochs')}")
    history = []
    best_state = None
    best_val = -float("inf")
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    run_info = {
        "run_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": config.get("model", config.get("model_name", "unknown")),
        "hyperparameters": config,
        "best_val_f1": None,
        "epochs_trained": 0,
        "history": [],
        "test": None,
    }

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)   # NEW: val

        if scheduler is not None:
            scheduler.step()

        epoch_row = {
            "epoch": epoch,
            "train": {
                k: float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else v
                for k, v in train_metrics.items()
                if k not in {"n", "tn", "fp", "fn", "tp"}
            },
            "val": {  # NEW: this matches your JSON exactly
                k: float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else v
                for k, v in val_metrics.items()
                if k not in {"n", "tn", "fp", "fn", "tp"}
            },
            "time_sec": round(time.time() - t0, 2),
        }
        history.append(epoch_row)
        
        current = val_metrics.get(monitor) # get f1
        if current is None or np.isnan(current):
            current = val_metrics["accuracy"]

        improved = current > best_val
        if improved:
            patience_counter = 0
            best_val = current
            best_epoch = epoch
            #best_state = copy.deepcopy(model.state_dict())
            #torch.save({
            #    "model_state_dict": best_state,
            #    "config": config,
            #    "epoch": epoch,
            #    "best_metric": best_val,
            #}, best_path)
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print("Model improved")

        else:
            patience_counter += 1
            print(f"patience {patience_counter}/{patience}")

        
        run_info["epochs_trained"] = epoch
        run_info["best_val_f1"] = float(best_val) if best_val != -float("inf") else None
        run_info["history"] = history

        if verbose:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss {train_metrics['loss']:.4f} f1 {train_metrics['f1']:.4f} acc {train_metrics['accuracy']:.4f} | "
                f"val loss {val_metrics['loss']:.4f} f1 {val_metrics['f1']:.4f} acc {val_metrics['accuracy']:.4f} | "
            )

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
            break

    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), f"{SAVING_FILE_NAME}.pth")   
        model.load_state_dict(best_model_state)

    final_test = evaluate(model, test_loader, criterion, device)
    run_info["test"] = {
        k: float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else v
        for k, v in final_test.items()
        if k not in {"n", "tn", "fp", "fn", "tp"}
    }

    save_json(run_info, os.path.join(run_dir, "results.json"))
    return run_info, model
    
# CNN from scratch USAGE only ---------------------------------------------------------------------------------------------
def build_model():
    model = SimpleCNN3().to(device)
    return model

def build_loss_and_optimizer(model, train_df, lr=1e-3):
    class_counts = train_df["label"].value_counts().sort_index()
    neg = int(class_counts.get(0, 0))
    pos = int(class_counts.get(1, 0))

    # pondérer la classe positive pour garder rééquilibrage
    # poids classe 0 = 1.0 ; poids classe 1 = neg / pos
    class_weights = torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights) # classif binaire avec 2 logits (to use pipeline.ipynb)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return criterion, optimizer

# helper pour lancer une expérience pipeline.py sur un jeu de loaders
def run_experiment(train_dataset, val_dataset, train_loader, val_loader, test_dataset, test_loader, run_dir, train_df, experiment_name):
    model = build_model()
    criterion, optimizer = build_loss_and_optimizer(model, train_df, lr=1e-3)

    config = {
        "model": "SimpleCNN3",
        "batch_size": BATCH_SIZE,
        "epochs": 10,
        "learning_rate": 1e-3,
        "loss": "CrossEntropyLoss",
        "patch_size": PATCH_SIZE,
        "seed": SEED,
        "experiment": experiment_name,
    }

    pipeline.SAVING_FILE_NAME = os.path.join(run_dir, "best_model")

    run_info, best_model = fit_binary_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        run_dir=run_dir,
        config=config,
        patience=5,
        monitor="f1",
    )

    y_true, scores = predict_probs(best_model, test_loader, device)
    preds05 = (scores >= 0.5).astype(int)
    preds07 = (scores >= 0.7).astype(int)

    print(experiment_name)
    print("test @0.5")
    print(confusion_matrix(y_true, preds05))
    print(classification_report(y_true, preds05, target_names=["Negative", "Positive"]))

    return {
        "run_info": run_info,
        "best_model": best_model,
        "y_true": y_true,
        "scores": scores,
        "preds05": preds05,
        "preds07": preds07,
    }
# ---------------------------------------------------------------------------------------------------------------------------

def eval_clip(model, processor, loader, prompts, device, max_batches=100):
    """Evaluate clip model on a dataloader and return metrics."""
    model.eval()
    all_positive_probs = []
    all_labels = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(loader)):
            if i >= max_batches: break
            
            images = images.to(device)
            inputs = processor(
                text=prompts, 
                images=images,  
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # size (B, len(prompts))
            #print("logits",logits_per_image,logits_per_image.shape)
            probs = torch.softmax(logits_per_image,dim=1)
            #print("probs",probs,probs.shape)
            pos_prob = probs[:,0] # the first column is for trees (second is no tree)

            all_positive_probs.extend(pos_prob.cpu().numpy())
            all_labels.extend(labels.numpy())
    y_prob = np.array(all_positive_probs)
    y_true = np.array(all_labels)
    metrics = compute_binary_metrics(y_true, y_prob)
    return metrics, y_prob, y_true