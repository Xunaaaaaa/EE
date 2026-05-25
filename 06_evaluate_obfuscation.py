"""
Evaluate both models (TF-IDF+SVM and CodeBERT) on obfuscated malicious scripts
at three levels: token, string, and encoding.

Generates:
  - obfuscation_detection_rate.png  (line chart: detection rate vs level)
  - obfuscation_metrics.csv         (detailed metrics per level per model)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score
from config import MODELS_DIR, RESULTS_DIR, DATA_DIR

OBF_DIR = DATA_DIR / "obfuscation"

LEVELS = ["clean", "case", "token", "concat", "string", "whitespace", "layered", "encoding", "mixed"]
LEVEL_LABELS = [
    "Clean", "Case\n(Randomize)", "Token\n(Backtick)", "Concat\n(Split+Join)",
    "String\n(Reorder)", "Whitespace\n(Comments)", "Layered\n(Token+String)",
    "Encoding\n(Base64)", "Mixed\n(Embedded)",
]

CODEBERT_MODEL_DIR = RESULTS_DIR / "codebert_model"
CODEBERT_MAX_LENGTH = 512
CODEBERT_BATCH_SIZE = 4


def load_svm():
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    model = joblib.load(MODELS_DIR / "svm_model.joblib")
    return vectorizer, model


def predict_svm(vectorizer, model, texts):
    X = vectorizer.transform(texts)
    return model.predict(X)


def load_codebert(device):
    tokenizer = RobertaTokenizer.from_pretrained(CODEBERT_MODEL_DIR)
    model = RobertaForSequenceClassification.from_pretrained(CODEBERT_MODEL_DIR)
    model.to(device)
    model.eval()
    return tokenizer, model


def predict_codebert(tokenizer, model, texts, device):
    all_preds = []
    for i in range(0, len(texts), CODEBERT_BATCH_SIZE):
        batch_texts = texts[i:i + CODEBERT_BATCH_SIZE]
        encoding = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=CODEBERT_MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_preds)


def load_obfuscated_data(level):
    if level == "clean":
        path = OBF_DIR / "clean_malicious.csv"
    else:
        path = OBF_DIR / f"obfuscated_{level}.csv"
    if level == "mixed":
        df = pd.read_csv(path, escapechar='\\')
    else:
        df = pd.read_csv(path)
    df["script_text"] = df["script_text"].fillna("")
    return df["script_text"].tolist()


def plot_detection_rate(results_df):
    fig, ax = plt.subplots(figsize=(13, 6))

    svm_rates = results_df[results_df["model"] == "TF-IDF + SVM"]["detection_rate"].values
    cb_rates = results_df[results_df["model"] == "CodeBERT"]["detection_rate"].values

    x = np.arange(len(LEVELS))
    ax.plot(x, svm_rates, 'o-', color="#4C72B0", linewidth=2, markersize=8, label="TF-IDF + SVM")
    ax.plot(x, cb_rates, 's-', color="#DD8452", linewidth=2, markersize=8, label="CodeBERT")

    for i in range(len(LEVELS)):
        ax.annotate(f"{svm_rates[i]:.1%}", (x[i], svm_rates[i]),
                    textcoords="offset points", xytext=(0, 12), ha='center', fontsize=9, color="#4C72B0")
        ax.annotate(f"{cb_rates[i]:.1%}", (x[i], cb_rates[i]),
                    textcoords="offset points", xytext=(0, -18), ha='center', fontsize=9, color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_LABELS)
    ax.set_ylabel("Detection Rate (Recall for Malicious Class)")
    ax.set_title("Model Robustness Under Obfuscation")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "obfuscation_detection_rate.png", dpi=150)
    plt.close(fig)
    print("  Saved obfuscation_detection_rate.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading SVM model...")
    svm_vec, svm_model = load_svm()

    print("Loading CodeBERT model...")
    cb_tokenizer, cb_model = load_codebert(device)

    rows = []

    for level in LEVELS:
        print(f"\n--- Level: {level} ---")
        texts = load_obfuscated_data(level)
        true_labels = np.ones(len(texts), dtype=int)

        svm_preds = predict_svm(svm_vec, svm_model, texts)
        svm_detection = (svm_preds == 1).mean()
        print(f"  SVM detection rate: {svm_detection:.4f}")
        rows.append({
            "model": "TF-IDF + SVM",
            "level": level,
            "detection_rate": svm_detection,
            "f1_malicious": f1_score(true_labels, svm_preds, pos_label=1, zero_division=0),
            "precision": precision_score(true_labels, svm_preds, pos_label=1, zero_division=0),
            "recall": recall_score(true_labels, svm_preds, pos_label=1, zero_division=0),
            "n_scripts": len(texts),
        })

        cb_preds = predict_codebert(cb_tokenizer, cb_model, texts, device)
        cb_detection = (cb_preds == 1).mean()
        print(f"  CodeBERT detection rate: {cb_detection:.4f}")
        rows.append({
            "model": "CodeBERT",
            "level": level,
            "detection_rate": cb_detection,
            "f1_malicious": f1_score(true_labels, cb_preds, pos_label=1, zero_division=0),
            "precision": precision_score(true_labels, cb_preds, pos_label=1, zero_division=0),
            "recall": recall_score(true_labels, cb_preds, pos_label=1, zero_division=0),
            "n_scripts": len(texts),
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_DIR / "obfuscation_metrics.csv", index=False)
    print(f"\nSaved obfuscation_metrics.csv")

    print("\nGenerating charts...")
    plot_detection_rate(results_df)

    print("\n" + results_df.to_string(index=False))
    print(f"\nAll outputs saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
