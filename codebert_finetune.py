"""
CodeBERT fine-tuning for PowerShell malicious script detection.

USAGE (Google Colab):
  1. Upload train.csv, val.csv, test.csv to Colab (or mount Google Drive)
  2. Upload this script to Colab
  3. In a Colab cell, run: !pip install transformers torch scikit-learn pandas
  4. In another cell, run: !python codebert_finetune.py --data_dir /content --output_dir /content/results
  5. Download codebert_predictions.csv and codebert_metrics.json from /content/results

USAGE (local, inside the repo): python codebert_finetune.py
  Defaults to config.py's per-seed directories and seed (EE_SEED env var).
"""

import argparse
import json
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, balanced_accuracy_score, confusion_matrix
)
from pathlib import Path

try:  # running inside the repo: use config.py's per-seed paths
    from config import PROCESSED_DIR, RESULTS_DIR, RANDOM_SEED
    DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_SEED = str(PROCESSED_DIR), str(RESULTS_DIR), RANDOM_SEED
except ImportError:  # standalone (e.g. Colab)
    DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_SEED = "data/processed", "results", 42


MODEL_NAME = "microsoft/codebert-base"
EPOCHS = 4
LR = 2e-5
BATCH_SIZE = 16
MAX_LENGTH = 512
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01


class ScriptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def classification_metrics(y_true, y_pred):
    """Same metric set as 03_train_svm.py (malicious = positive class)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_benign": float(f1_score(y_true, y_pred, pos_label=0)),
        "f1_malicious": float(f1_score(y_true, y_pred, pos_label=1)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "precision_benign": float(precision_score(y_true, y_pred, pos_label=0)),
        "precision_malicious": float(precision_score(y_true, y_pred, pos_label=1)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "recall_benign": float(recall_score(y_true, y_pred, pos_label=0)),
        "recall_malicious": float(recall_score(y_true, y_pred, pos_label=1)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)},
    }


def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights):
    model.train()
    total_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directory containing train.csv, val.csv, test.csv")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save predictions and metrics")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    set_seed(args.seed)
    batch_size = args.batch_size

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")
    hardware = {"device": str(device), "torch_version": torch.__version__}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        hardware.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_vram_gib": round(props.total_memory / 1024 ** 3, 2),
            "cuda_version": torch.version.cuda,
        })
        print(f"GPU: {hardware['gpu_name']} ({hardware['gpu_vram_gib']} GiB VRAM), CUDA {torch.version.cuda}")
    print(f"Batch size: {batch_size}, seed: {args.seed}")

    print("Loading data...")
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    train_df["script_text"] = train_df["script_text"].fillna("")
    val_df["script_text"] = val_df["script_text"].fillna("")
    test_df["script_text"] = test_df["script_text"].fillna("")

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("Loading tokenizer and model...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    train_dataset = ScriptDataset(
        train_df["script_text"].tolist(), train_df["label"].tolist(),
        tokenizer, MAX_LENGTH
    )
    val_dataset = ScriptDataset(
        val_df["script_text"].tolist(), val_df["label"].tolist(),
        tokenizer, MAX_LENGTH
    )
    test_dataset = ScriptDataset(
        test_df["script_text"].tolist(), test_df["label"].tolist(),
        tokenizer, MAX_LENGTH
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    n_benign = (train_df["label"] == 0).sum()
    n_malicious = (train_df["label"] == 1).sum()
    weight_benign = len(train_df) / (2 * n_benign)
    weight_malicious = len(train_df) / (2 * n_malicious)
    class_weights = torch.tensor([weight_benign, weight_malicious], dtype=torch.float32)
    print(f"Class weights: benign={weight_benign:.3f}, malicious={weight_malicious:.3f}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\nTraining for {EPOCHS} epochs ({total_steps} steps, {warmup_steps} warmup)...")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_start = time.perf_counter()
    train_time = 0.0  # pure training (tokenisation + forward/backward); excludes per-epoch val monitoring
    epoch_log = []

    for epoch in range(EPOCHS):
        t0 = time.perf_counter()
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, device, class_weights)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0
        train_time += epoch_time

        val_labels, val_preds, _ = evaluate(model, val_loader, device)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        epoch_log.append({"epoch": epoch + 1, "loss": float(avg_loss), "val_f1_macro": float(val_f1),
                          "train_seconds": round(epoch_time, 2)})

        print(f"  Epoch {epoch + 1}/{EPOCHS}: loss={avg_loss:.4f}, val_f1={val_f1:.4f}, time={epoch_time:.0f}s")

    wall_time = time.perf_counter() - train_start
    print(f"Training complete in {train_time:.0f}s ({wall_time:.0f}s including per-epoch validation)")
    if device.type == "cuda":
        hardware["peak_vram_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)

    model_save_dir = output_dir / "codebert_model"
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    print(f"Saved fine-tuned model to {model_save_dir}")

    # End-to-end classification: tokenisation (in the DataLoader) + forward pass.
    print("\nEvaluating on test set...")
    t0 = time.perf_counter()
    test_labels, test_preds, test_probs = evaluate(model, test_loader, device)
    inference_time = time.perf_counter() - t0

    metrics = {
        "model": "CodeBERT",
        "seed": args.seed,
        **classification_metrics(test_labels, test_preds),
        "train_time_seconds": round(train_time, 2),
        "train_wall_time_incl_validation_seconds": round(wall_time, 2),
        "inference_time_seconds": round(inference_time, 4),
        "inference_per_script_ms": round((inference_time / len(test_labels)) * 1000, 4),
        "n_test_scripts": len(test_labels),
        "batch_size": batch_size,
        "learning_rate": LR,
        "epochs": EPOCHS,
        "max_length": MAX_LENGTH,
        "class_weights": {"benign": round(float(weight_benign), 6), "malicious": round(float(weight_malicious), 6)},
        "train_class_counts": {"benign": int(n_benign), "malicious": int(n_malicious)},
        "epoch_log": epoch_log,
        "hardware": hardware,
    }

    print(f"\nTest Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    preds_df = pd.DataFrame({
        "source_file": test_df.get("source_file"),
        "text_sha256": test_df.get("text_sha256"),
        "true_label": test_labels,
        "predicted_label": test_preds,
        "malicious_probability": test_probs,
    })
    preds_df.to_csv(output_dir / "codebert_predictions.csv", index=False)

    with open(output_dir / "codebert_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved predictions and metrics to {output_dir}")


if __name__ == "__main__":
    main()
